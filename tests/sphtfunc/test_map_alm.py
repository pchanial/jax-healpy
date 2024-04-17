from pathlib import Path
from typing import Callable

import healpy as hp
import jax.numpy as jnp
import numpy as np
import pytest
from s2fft.sampling.s2_samples import flm_2d_to_hp, flm_hp_to_2d

import jax_healpy as jhp


@pytest.fixture(scope='session')
def cla(data_path: Path) -> np.ndarray:
    return hp.read_cl(data_path / 'cl_wmap_band_iqumap_r9_7yr_W_v4_udgraded32_II_lmax64_rmmono_3iter.fits')


@pytest.fixture(scope='session')
def synthesized_map(cla: np.ndarray) -> np.ndarray:
    nside = 32
    lmax = 64
    fwhm_deg = 7.0
    seed = 12345
    np.random.seed(seed)
    return hp.synfast(
        cla,
        nside,
        lmax=lmax,
        pixwin=False,
        fwhm=np.radians(fwhm_deg),
        new=False,
    )


@pytest.mark.parametrize('lmax', [None, 63])
@pytest.mark.parametrize('healpy_ordering', [False, True])
def test_map2alm(synthesized_map: np.ndarray, lmax: int | None, healpy_ordering: bool) -> None:
    nside = hp.npix2nside(synthesized_map.size)
    actual_flm = jhp.map2alm(synthesized_map, lmax=lmax, iter=0, healpy_ordering=healpy_ordering)

    expected_flm = hp.map2alm(synthesized_map, lmax=lmax, iter=0)
    if not healpy_ordering:
        L = 3 * nside if lmax is None else lmax + 1
        expected_flm = flm_hp_to_2d(expected_flm, L)
    np.testing.assert_allclose(actual_flm, expected_flm, atol=1e-14)


@pytest.mark.parametrize('lmax', [None, 7])
@pytest.mark.parametrize('healpy_ordering', [False, True])
def test_alm2map(flm_generator: Callable[[...], np.ndarray], lmax: int | None, healpy_ordering: bool) -> None:
    nside = 4
    if lmax is None:
        L = 3 * nside
    else:
        L = lmax + 1
    flm = flm_generator(L=L, spin=0, reality=True, healpy_ordering=healpy_ordering)
    actual_map = jhp.alm2map(flm, nside, lmax=lmax, healpy_ordering=healpy_ordering)

    if not healpy_ordering:
        flm = flm_2d_to_hp(flm, L)
    expected_map = hp.alm2map(flm, nside, lmax=lmax, pol=False)

    np.testing.assert_allclose(actual_map, expected_map, atol=1e-14)


def test_alm2map_scalar_error() -> None:
    with pytest.raises(ValueError):
        _ = jhp.alm2map(jnp.array(0.0 + 0j), nside=1)


def test_map2alm_scalar_error() -> None:
    with pytest.raises(ValueError):
        _ = jhp.map2alm(jnp.array(0.0), iter=0)


def test_alm2map_invalid_ndim_error() -> None:
    with pytest.raises(ValueError):
        _ = jhp.alm2map(jnp.array([[0.0 + 0j]]), nside=1)


def test_alm2map_invalid_ndim_healpy_ordering_error() -> None:
    with pytest.raises(ValueError):
        _ = jhp.alm2map(jnp.array([[[0.0 + 0j]]]), nside=1, healpy_ordering=True)


def test_map2alm_invalid_ndim_error() -> None:
    with pytest.raises(ValueError):
        _ = jhp.map2alm(jnp.array([[[0.0]]]), iter=0)
