import os
import numpy as np
import sinter
import matplotlib.pyplot as plt
from stimbposd import sinter_decoders

from src.generateBBcode import *   

PS = [1e-3, 1.5e-3, 2e-3, 3e-3, 4e-3, 6e-3, 8e-3]
ROUNDS = 6
K = 12
CSV = "bb72_bposd.csv"


def make_tasks():
    for p in PS:
        l = 6
        m = 6

        A_vars = [[0,3], [1,1], [1,2]]
        B_vars = [[1,3], [0,1], [0,2]]
        code = BivBic(l = l,m = m,A_vars = A_vars,B_vars = B_vars, p = p, rounds = ROUNDS)          # adapt to your signature
        yield sinter.Task(
            circuit=code.circuit,
            json_metadata={"p": p, "rounds": ROUNDS, "k": K, "basis": "Z"},
        )


def collect():
    sinter.collect(
        num_workers=os.cpu_count(),
        tasks=list(make_tasks()),
        decoders=["bposd"],
        custom_decoders=sinter_decoders(),
        max_shots=200_000,
        max_errors=500,
        count_observable_error_combos=True,
        count_detection_events=True,
        print_progress=True,
        save_resume_filepath=CSV,
    )


def per_cycle(P_L, nc):
    return 1 - (1 - P_L) ** (1 / nc)


def analyse(stats):
    rows = []
    for s in sorted(stats, key=lambda s: s.json_metadata["p"]):
        n = s.shots - s.discards
        nc = s.json_metadata["rounds"]
        f = sinter.fit_binomial(num_shots=n, num_hits=s.errors,
                                max_likelihood_factor=1e3)

        obs = sum(c * key.split("=", 1)[1].count("E")
                  for key, c in s.custom_counts.items()
                  if key.startswith("obs_mistake_mask="))
        det = (s.custom_counts["detection_events"]
               / s.custom_counts["detectors_checked"])

        rows.append(dict(
            p=s.json_metadata["p"], shots=n, errors=s.errors,
            P_L=f.best, pl=per_cycle(f.best, nc),
            pl_lo=per_cycle(f.low, nc), pl_hi=per_cycle(f.high, nc),
            obs_per_fail=obs / max(s.errors, 1), det_frac=det,
        ))

    print(f"{'p':>8} {'shots':>8} {'errs':>6} {'P_L':>9} {'p_L':>9} "
          f"{'obs/fail':>9} {'det':>7}")
    for r in rows:
        print(f"{r['p']:8.4f} {r['shots']:8d} {r['errors']:6d} "
              f"{r['P_L']:9.4g} {r['pl']:9.4g} "
              f"{r['obs_per_fail']:9.2f} {r['det_frac']:7.3f}")
    return rows


def plot(stats, rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    sinter.plot_error_rate(
        ax=axes[0], stats=stats,
        x_func=lambda s: s.json_metadata["p"],
        group_func=lambda s: f"[[72,12,6]], {s.json_metadata['rounds']}r, {s.decoder}",
    )
    p = np.array(PS)
    axes[0].plot(p, K * p, "k--", label="break-even $kp$")
    axes[0].set_ylabel("$P_L$ per shot")

    x = [r["p"] for r in rows]
    y = np.array([r["pl"] for r in rows])
    lo = np.array([r["pl_lo"] for r in rows])
    hi = np.array([r["pl_hi"] for r in rows])
    axes[1].errorbar(x, y, yerr=[y - lo, hi - y], marker="o", capsize=3,
                     label="BP-OSD")
    axes[1].set_ylabel("$p_L$ per cycle")

    for ax in axes:
        ax.loglog()
        ax.grid(which="both", alpha=0.3)
        ax.set_xlabel("physical error rate $p$")
        ax.legend()

    fig.tight_layout()
    fig.savefig("bb72_bposd.png", dpi=150)
    #plt.show()


if __name__ == "__main__":
    collect()                                    # no-op if CSV is complete
    stats = sinter.read_stats_from_csv_files(CSV)
    plot(stats, analyse(stats))