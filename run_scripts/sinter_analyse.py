#!/usr/bin/env python
"""BP-OSD logical error rate sweep for bivariate bicycle codes."""

import argparse, glob, os
import numpy as np
import sinter
from stimbposd import SinterDecoder_BPOSD

from src.generate_BBcode import *  

# (l, m, A, B) from Bravyi et al., Nature 627, 778 (2024).
CODES = {
    "bb72":  dict(l=6,  m=6,  A_vars=[[0,3], [1,1],  [1,2]], B_vars=[[1,3], [0,1],  [0,2]], n=72,  k=12, d=6),
    "bb90":  dict(l=15,  m=3,  A_vars=[[0,9], [1,1],  [1,2]], B_vars=[[0,0], [0,2],  [0,7]], n=90,  k=8, d=10),
    "bb108":  dict(l=9,  m=6,  A_vars=[[0,3], [1,1],  [1,2]], B_vars=[[1,3], [0,1],  [0,2]], n=108,  k=8, d=10)
}


PS = {
    "bb72":  [0.001, 0.002, 0.003, 0.004, 0.006, 0.008],
    "bb90":  [0.002, 0.003, 0.004, 0.005, 0.006, 0.008],
    "bb108": [0.002, 0.003, 0.004, 0.005, 0.006, 0.008],
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", nargs="+", default=list(CODES), choices=list(CODES))
    ap.add_argument("--noises", type=float, nargs="+", default=None,
                    help="override per-code default p list")
    ap.add_argument("--rounds", type=int, default=None,
                    help="default: N_c = d, matching the paper")
    ap.add_argument("--basis", default="Z", choices=["X", "Z"])
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--max_shots", type=int, default=200_000)
    ap.add_argument("--max_errors", type=int, default=500)
    ap.add_argument("--osd_order", type=int, default=10)
    ap.add_argument("--max_bp_iters", type=int, default=30)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--rate", default="cycle", choices=["cycle", "shot"])
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--merge", nargs="+", default=None,
                    help="glob(s) of CSVs to analyse/plot without collecting")
    return ap.parse_args()


def make_tasks(args):
    for name in args.codes:
        cfg = CODES[name]
        nc = args.rounds or cfg["d"]
        for p in (args.noises or PS[name]):
            code = BivBic(l=cfg["l"], m=cfg["m"], A_vars=cfg["A_vars"], B_vars=cfg["B_vars"], p = p, rounds = nc)
            yield sinter.Task(
                circuit=code.circuit,
                json_metadata={"code": name, "n": cfg["n"], "k": cfg["k"],
                               "d": cfg["d"], "rounds": nc, "p": p,
                               "basis": args.basis},
            )


def collect(args):
    sinter.collect(
        num_workers=args.workers,
        tasks=list(make_tasks(args)),
        decoders=["bposd"],
        custom_decoders={"bposd": SinterDecoder_BPOSD(
            max_bp_iters=args.max_bp_iters, osd_order=args.osd_order)},
        max_shots=args.max_shots,
        max_errors=args.max_errors,
        count_observable_error_combos=True,
        count_detection_events=True,
        print_progress=True,
        save_resume_filepath=args.csv,
    )


def rate(P, nc, how):
    """Convert block-per-shot probability to the chosen convention."""
    return P if how == "shot" else 1 - (1 - P) ** (1 / nc)


def analyse(stats, how):
    rows = []
    for s in sorted(stats, key=lambda s: (s.json_metadata["n"],
                                          s.json_metadata["p"])):
        md = s.json_metadata
        n_shots = s.shots - s.discards
        nc = md["rounds"]
        f = sinter.fit_binomial(num_shots=n_shots, num_hits=s.errors,
                                max_likelihood_factor=1e3)

        obs = sum(c * key.split("=", 1)[1].count("E")
                  for key, c in s.custom_counts.items()
                  if key.startswith("obs_mistake_mask="))
        det = (s.custom_counts["detection_events"]
               / s.custom_counts["detectors_checked"])

        rows.append(dict(
            code=md["code"], label=f"[[{md['n']},{md['k']},{md['d']}]]",
            k=md["k"], rounds=nc, p=md["p"], shots=n_shots, errors=s.errors,
            y=rate(f.best, nc, how),
            lo=rate(f.low, nc, how), hi=rate(f.high, nc, how),
            obs_per_fail=obs / max(s.errors, 1), det_frac=det,
        ))

    hdr = f"{'code':>8} {'p':>7} {'shots':>9} {'errs':>6} " \
          f"{'p_L':>10} {'lo':>10} {'hi':>10} {'obs/fail':>9} {'det':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['code']:>8} {r['p']:7.4f} {r['shots']:9d} {r['errors']:6d} "
              f"{r['y']:10.3e} {r['lo']:10.3e} {r['hi']:10.3e} "
              f"{r['obs_per_fail']:9.2f} {r['det_frac']:7.3f}")
    return rows


def plot(rows, how, out="bb_sweep.png"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    by_code = {}
    for r in rows:
        by_code.setdefault(r["label"], []).append(r)

    for label, rs in sorted(by_code.items(), key=lambda kv: rs_n(kv[1])):
        rs.sort(key=lambda r: r["p"])
        x = np.array([r["p"] for r in rs])
        y = np.array([r["y"] for r in rs])
        lo = np.array([r["lo"] for r in rs])
        hi = np.array([r["hi"] for r in rs])
        ax.errorbar(x, y, yerr=[y - lo, hi - y], marker="o", capsize=3,
                    label=f"{label}, $N_c$={rs[0]['rounds']}")

    # break-even: k bare qubits over the same exposure as the data points
    p = np.logspace(np.log10(min(r["p"] for r in rows)),
                    np.log10(max(r["p"] for r in rows)), 100)
    k = rows[0]["k"]
    if how == "cycle":
        be, be_lbl = 1 - (1 - p) ** k, f"break-even, $k$={k} bare qubits/cycle"
    else:
        nc = rows[0]["rounds"]
        be = 1 - (1 - p) ** (k * nc)
        be_lbl = f"break-even, $k$={k} bare qubits $\\times$ {nc} cycles"
    ax.plot(p, be, "k--", lw=1, label=be_lbl)

    ax.loglog()
    ax.grid(which="both", alpha=0.3)
    ax.set_xlabel("physical error rate $p$")
    ax.set_ylabel("$p_L$ per cycle" if how == "cycle" else "$P_L$ per shot")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def rs_n(rs):
    return rs[0]["k"], rs[0]["rounds"]


if __name__ == "__main__":
    args = parse_args()

    if args.merge:
        files = [f for g in args.merge for f in glob.glob(g)]
        stats = sinter.read_stats_from_csv_files(*files)
    else:
        tag = "-".join(args.codes)
        args.csv = args.csv or f"results/{tag}_{args.basis}_bposd.csv"
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        collect(args)
        stats = sinter.read_stats_from_csv_files(args.csv)

    rows = analyse(stats, args.rate)
    if args.plot:
        plot(rows, args.rate)