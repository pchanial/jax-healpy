import numpy as np
import pytest
from numpy.testing import assert_allclose

import jax_healpy as hp


@pytest.mark.parametrize(
    'z, iphi',
    [
        (1, 0),
        (0.999, 0),
        (0.999, 1),
        (0.999, 2),
        (0.999, 3),
        (0.999, 4),
        (0.98, 0),
        (0.98, 1),
        (0.98, 2),
        (0.98, 3),
        (0.98, 4),
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (-0.98, 0),
        (-0.98, 1),
        (-0.98, 2),
        (-0.98, 3),
        (-0.98, 4),
        (-0.999, 0),
        (-0.999, 1),
        (-0.999, 2),
        (-0.999, 3),
        (-0.999, 4),
        (-1, 0),
        (-1, 1),
        (-1, 2),
        (-1, 3),
        (-1, 4),
    ],
)
@pytest.mark.parametrize('lonlat', [False, True])
def test_ang2vec2ang(z: float, iphi: int, lonlat: bool) -> None:
    # test that we get the same results as healpy
    theta = np.arccos(z)
    phi = iphi / 5 * 2 * np.pi
    actual_theta, actual_phi = hp.vec2ang(hp.ang2vec(theta, phi, lonlat=lonlat), lonlat=lonlat)
    assert_allclose(actual_theta, theta, rtol=1e-14, atol=1e-15)
    assert_allclose(actual_phi, phi, rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize('lonlat', [False, True])
def test_ang2vec_array(lonlat: bool) -> None:
    theta = np.array([np.pi / 4, 3 * np.pi / 4])
    phi = np.array([np.pi / 2, 3 * np.pi / 2])
    vec0 = hp.ang2vec(theta[0], phi[0], lonlat=lonlat)
    assert vec0.shape == (3,)
    vec1 = hp.ang2vec(theta[1], phi[1], lonlat=lonlat)
    vec = hp.ang2vec(theta, phi, lonlat=lonlat)
    assert vec.shape == (2, 3)
    assert_allclose(vec, np.array([vec0, vec1]), rtol=1e-14)


@pytest.mark.parametrize('lonlat', [False, True])
def test_vec2ang_array(lonlat: bool) -> None:
    vec = np.array([[1, 2, 3], [-1, 2, -1]])
    theta0, phi0 = hp.vec2ang(vec[0], lonlat=lonlat)
    assert theta0.shape == (1,)
    assert phi0.shape == (1,)
    theta1, phi1 = hp.vec2ang(vec[1], lonlat=lonlat)
    theta, phi = hp.vec2ang(vec, lonlat=lonlat)
    assert theta.shape == (2,)
    assert phi.shape == (2,)
    assert_allclose(theta, np.array([theta0[0], theta1[0]]), rtol=1e-14)
    assert_allclose(phi, np.array([phi0[0], phi1[0]]), rtol=1e-14)
