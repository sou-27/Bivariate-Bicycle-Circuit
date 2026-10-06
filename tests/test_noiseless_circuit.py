from src.generate_BBcode import *
import pytest
import numpy as np
import galois


def test_commutation():
    l = 6
    m = 6

    A_vars = [[0,3], [1,1], [1,2]]
    B_vars = [[1,3], [0,1], [0,2]]

    code = BivBic(l,m,A_vars,B_vars)

    HX = code.HX
    HZ = code.HZ

    GF2 = galois.GF(2)

    commutation = HX@HZ.T
    
    mat_zero = GF2.Zeros(commutation.shape)

    assert np.array_equal(commutation,mat_zero), "Stabilizer matrices don't commute!"


def test_noiseless_circuit():
    l = 6
    m = 6

    A_vars = [[0,3], [1,1], [1,2]]
    B_vars = [[1,3], [0,1], [0,2]]

    code = BivBic(l,m,A_vars,B_vars)

    sampler = code.circuit.compile_detector_sampler()
    nshots = 1000   

    _, observable_flips = sampler.sample(shots = nshots, separate_observables = True)

    assert not np.all(observable_flips), "Cirucit is not noiseless"
    
  