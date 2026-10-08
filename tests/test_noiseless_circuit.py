from src.generate_BBcode import *
import pytest
import numpy as np
import galois


def test_commutation():
    """
    Tests to see if computed H_X and H_Z satisfy required orthogonality condition H_X . H_Z^T = 0
    """
    l = 6
    m = 6

    A_vars = [[0,3], [1,1], [1,2]]
    B_vars = [[1,3], [0,1], [0,2]]

    p = 0.01
    rounds = 10

    code = BivBic(l = l,m = m,A_vars = A_vars,B_vars = B_vars, p = p, rounds = rounds)

    HX = code.HX
    HZ = code.HZ

    GF2 = galois.GF(2)

    commutation = HX@HZ.T
    
    mat_zero = GF2.Zeros(commutation.shape)

    assert np.array_equal(commutation,mat_zero), "Stabilizer matrices don't commute!"


def test_noiseless_circuit():
    """
    In deterministic limit p = 0, logical observables should never flip.
    """
    l = 6
    m = 6

    A_vars = [[0,3], [1,1], [1,2]]
    B_vars = [[1,3], [0,1], [0,2]]

    rounds = 10
    p = 0

    code = BivBic(l = l,m = m,A_vars = A_vars,B_vars = B_vars, p = p, rounds = rounds)

    sampler = code.circuit.compile_detector_sampler()
    nshots = 1000   

    _, observable_flips = sampler.sample(shots = nshots, separate_observables = True)

    assert not np.all(observable_flips), "Circuit is not noiseless"
    
  