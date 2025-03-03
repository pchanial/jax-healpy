"""
=====================================================
pixelfunc.py : Healpix pixelization related functions
=====================================================

This module provides functions related to Healpix pixelization scheme.

conversion from/to sky coordinates
----------------------------------

- :func:`pix2ang` converts pixel number to angular coordinates
- :func:`pix2vec` converts pixel number to unit 3-vector direction
- :func:`ang2pix` converts angular coordinates to pixel number
- :func:`vec2pix` converts 3-vector to pixel number
- :func:`vec2ang` converts 3-vector to angular coordinates
- :func:`ang2vec` converts angular coordinates to unit 3-vector
- :func:`pix2xyf` converts pixel number to coordinates within face
- :func:`xyf2pix` converts coordinates within face to pixel number
- :func:`get_interp_weights` returns the 4 nearest pixels for given
  angular coordinates and the relative weights for interpolation
- :func:`get_all_neighbours` return the 8 nearest pixels for given
  angular coordinates

conversion between NESTED and RING schemes
------------------------------------------

- :func:`nest2ring` converts NESTED scheme pixel numbers to RING
  scheme pixel number
- :func:`ring2nest` converts RING scheme pixel number to NESTED
  scheme pixel number
- :func:`reorder` reorders a healpix map pixels from one scheme to another

nside/npix/resolution
---------------------

- :func:`nside2npix` converts healpix nside parameter to number of pixel
- :func:`npix2nside` converts number of pixel to healpix nside parameter
- :func:`nside2order` converts nside to order
- :func:`order2nside` converts order to nside
- :func:`nside2resol` converts nside to mean angular resolution
- :func:`nside2pixarea` converts nside to pixel area
- :func:`isnsideok` checks the validity of nside
- :func:`isnpixok` checks the validity of npix
- :func:`get_map_size` gives the number of pixel of a map
- :func:`get_min_valid_nside` gives the minimum nside possible for a given
  number of pixel
- :func:`get_nside` returns the nside of a map
- :func:`maptype` checks the type of a map (one map or sequence of maps)
- :func:`ud_grade` upgrades or degrades the resolution (nside) of a map

Masking pixels
--------------

- :const:`UNSEEN` is a constant value interpreted as a masked pixel
- :func:`mask_bad` returns a map with ``True`` where map is :const:`UNSEEN`
- :func:`mask_good` returns a map with ``False`` where map is :const:`UNSEEN`
- :func:`ma` returns a masked array as map, with mask given by :func:`mask_bad`

Map data manipulation
---------------------

- :func:`fit_dipole` fits a monopole+dipole on the map
- :func:`fit_monopole` fits a monopole on the map
- :func:`remove_dipole` fits and removes a monopole+dipole from the map
- :func:`remove_monopole` fits and remove a monopole from the map
- :func:`get_interp_val` computes a bilinear interpolation of the map
  at given angular coordinates, using 4 nearest neighbours
"""

from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from jaxtyping import Array, ArrayLike

__all__ = [
    'pix2ang',
    'ang2pix',
    'pix2vec',
    'vec2pix',
    'ang2vec',
    'vec2ang',
    # 'get_interp_weights',
    # 'get_interp_val',
    # 'get_all_neighbours',
    # 'max_pixrad',
    # 'nest2ring',
    # 'ring2nest',
    # 'reorder',
    # 'ud_grade',
    'UNSEEN',
    # 'mask_good',
    # 'mask_bad',
    # 'ma',
    # 'fit_dipole',
    # 'remove_dipole',
    # 'fit_monopole',
    # 'remove_monopole',
    'nside2npix',
    'npix2nside',
    'nside2order',
    'order2nside',
    'order2npix',
    'npix2order',
    'nside2resol',
    'nside2pixarea',
    'isnsideok',
    'isnpixok',
    # 'get_map_size',
    # 'get_min_valid_nside',
    # 'get_nside',
    'maptype',
    # 'ma_to_array',
]

# We are using 64-bit integer types.
# nside > 2**29 requires extended integer types.
MAX_NSIDE = 1 << 29
UNSEEN = -1.6375e30


def check_theta_valid(theta: ArrayLike) -> None:
    """Raises exception if theta is not within 0 and pi"""
    if (theta < 0).any() or (theta > np.pi + 1e-5).any():
        raise ValueError('THETA is out of range [0,𝝥]')


def check_nside(nside: int, nest: bool = False) -> None:
    """Raises exception is nside is not valid"""
    if not np.all(isnsideok(nside, nest=nest)):
        raise ValueError('%s is not a valid nside parameter (must be a power of 2, less than 2**30)' % str(nside))


def isnsideok(nside: int, nest: bool = False) -> bool:
    """Returns :const:`True` if nside is a valid nside parameter, :const:`False` otherwise.

    NSIDE needs to be a power of 2 only for nested ordering

    Parameters
    ----------
    nside : int, scalar or array-like
      integer value to be tested

    Returns
    -------
    ok : bool, scalar or array-like
      :const:`True` if given value is a valid nside, :const:`False` otherwise.

    Examples
    --------
    >>> import jax_healpy as hp
    >>> hp.isnsideok(13, nest=True)
    False

    >>> hp.isnsideok(13, nest=False)
    True

    >>> hp.isnsideok(32)
    True

    >>> hp.isnsideok([1, 2, 3, 4, 8, 16], nest=True)
    array([ True,  True, False,  True,  True,  True], dtype=bool)
    """
    # we use standard bithacks from http://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
    if hasattr(nside, '__len__'):
        if not isinstance(nside, np.ndarray):
            nside = np.asarray(nside)
        is_nside_ok = (nside == nside.astype(int)) & (nside > 0) & (nside <= MAX_NSIDE)
        if nest:
            is_nside_ok &= (nside.astype(int) & (nside.astype(int) - 1)) == 0
    else:
        is_nside_ok = nside == int(nside) and 0 < nside <= MAX_NSIDE
        if nest:
            is_nside_ok = is_nside_ok and (int(nside) & (int(nside) - 1)) == 0
    return is_nside_ok


def isnpixok(npix: int) -> bool:
    """Return :const:`True` if npix is a valid value for healpix map size, :const:`False` otherwise.

    Parameters
    ----------
    npix : int, scalar or array-like
      integer value to be tested

    Returns
    -------
    ok : bool, scalar or array-like
      :const:`True` if given value is a valid number of pixel, :const:`False` otherwise

    Examples
    --------
    >>> import jax_healpy as hp
    >>> hp.isnpixok(12)
    True

    >>> hp.isnpixok(768)
    True

    >>> hp.isnpixok([12, 768, 1002])
    array([ True,  True, False], dtype=bool)
    """
    nside = np.sqrt(np.asarray(npix) / 12.0)
    return nside == np.floor(nside)


def nside2npix(nside: int) -> int:
    """Give the number of pixels for the given nside.

    Parameters
    ----------
    nside : int
      healpix nside parameter

    Returns
    -------
    npix : int
      corresponding number of pixels

    Examples
    --------
    >>> import jax_healpy as hp
    >>> import numpy as np
    >>> hp.nside2npix(8)
    768

    >>> np.all([hp.nside2npix(nside) == 12 * nside**2 for nside in [2**n for n in range(12)]])
    True

    >>> hp.nside2npix(7)
    588
    """
    return 12 * nside * nside


def npix2nside(npix: int) -> int:
    """Give the nside parameter for the given number of pixels.

    Parameters
    ----------
    npix : int
      the number of pixels

    Returns
    -------
    nside : int
      the nside parameter corresponding to npix

    Notes
    -----
    Raise a ValueError exception if number of pixel does not correspond to
    the number of pixel of a healpix map.

    Examples
    --------
    >>> import jax_healpy as hp
    >>> hp.npix2nside(768)
    8

    >>> np.all([hp.npix2nside(12 * nside**2) == nside for nside in [2**n for n in range(12)]])
    True

    >>> hp.npix2nside(1000)
    Traceback (most recent call last):
        ...
    ValueError: Wrong pixel number (it is not 12*nside**2)
    """
    if not isnpixok(npix):
        raise ValueError('Wrong pixel number (it is not 12*nside**2)')
    return int(np.sqrt(npix / 12.0))


def nside2order(nside: int) -> int:
    """Give the resolution order for a given nside.

    Parameters
    ----------
    nside : int
      healpix nside parameter; an exception is raised if nside is not valid
      (nside must be a power of 2, less than 2**30)

    Returns
    -------
    order : int
      corresponding order where nside = 2**(order)

    Notes
    -----
    Raise a ValueError exception if nside is not valid.

    Examples
    --------
    >>> import jax_healpy as hp
    >>> import numpy as np
    >>> hp.nside2order(128)
    7

    >>> all(hp.nside2order(2**o) == o for o in range(30))
    True

    >>> hp.nside2order(7)
    Traceback (most recent call last):
        ...
    ValueError: 7 is not a valid nside parameter (must be a power of 2, less than 2**30)
    """
    check_nside(nside, nest=True)
    return len(f'{nside:b}') - 1


def order2nside(order: int) -> int:
    """Give the nside parameter for the given resolution order.

    Parameters
    ----------
    order : int
      the resolution order

    Returns
    -------
    nside : int
      the nside parameter corresponding to order

    Notes
    -----
    Raise a ValueError exception if order produces an nside out of range.

    Examples
    --------
    >>> import jax_healpy as hp
    >>> hp.order2nside(7)
    128

    >>> print(hp.order2nside(np.arange(8)))
    [  1   2   4   8  16  32  64 128]

    >>> hp.order2nside(31)
    Traceback (most recent call last):
        ...
    ValueError: 2147483648 is not a valid nside parameter (must be a power of 2, less than 2**30)
    """
    nside = 1 << order
    check_nside(nside, nest=True)
    return nside


def order2npix(order: int) -> int:
    """Give the number of pixels for the given resolution order.

    Parameters
    ----------
    order : int
      the resolution order

    Returns
    -------
    npix : int
      corresponding number of pixels

    Notes
    -----
    A convenience function that successively applies order2nside then nside2npix to order.

    Examples
    --------
    >>> import jax_healpy as hp
    >>> hp.order2npix(7)
    196608

    >>> print(hp.order2npix(np.arange(8)))
    [    12     48    192    768   3072  12288  49152 196608]

    >>> hp.order2npix(31)
    Traceback (most recent call last):
        ...
    ValueError: 2147483648 is not a valid nside parameter (must be a power of 2, less than 2**30)
    """
    nside = order2nside(order)
    npix = nside2npix(nside)
    return npix


def npix2order(npix: int) -> int:
    """Give the resolution order for the given number of pixels.

    Parameters
    ----------
    npix : int
      the number of pixels

    Returns
    -------
    order : int
      corresponding resolution order

    Notes
    -----
    A convenience function that successively applies npix2nside then nside2order to npix.

    Examples
    --------
    >>> import jax_healpy as hp
    >>> hp.npix2order(768)
    3

    >>> np.all([hp.npix2order(12 * 4**order) == order for order in range(12)])
    True

    >>> hp.npix2order(1000)
    Traceback (most recent call last):
        ...
    ValueError: Wrong pixel number (it is not 12*nside**2)
    """
    nside = npix2nside(npix)
    order = nside2order(nside)
    return order


def nside2resol(nside: int, arcmin=False) -> float:
    """Give approximate resolution (pixel size in radian or arcmin) for nside.

    Resolution is just the square root of the pixel area, which is a gross
    approximation given the different pixel shapes

    Parameters
    ----------
    nside : int
      healpix nside parameter, must be a power of 2, less than 2**30
    arcmin : bool
      if True, return resolution in arcmin, otherwise in radian

    Returns
    -------
    resol : float
      approximate pixel size in radians or arcmin

    Notes
    -----
    Raise a ValueError exception if nside is not valid.

    Examples
    --------
    >>> import jax_healpy as hp
    >>> hp.nside2resol(128, arcmin = True)  # doctest: +FLOAT_CMP
    27.483891294539248

    >>> hp.nside2resol(256)
    0.0039973699529159707

    >>> hp.nside2resol(7)
    0.1461895297066412
    """
    resol = np.sqrt(nside2pixarea(nside))

    if arcmin:
        resol = np.rad2deg(resol) * 60

    return resol


def nside2pixarea(nside: int, degrees=False) -> float:
    """Give pixel area given nside in square radians or square degrees.

    Parameters
    ----------
    nside : int
      healpix nside parameter, must be a power of 2, less than 2**30
    degrees : bool
      if True, returns pixel area in square degrees, in square radians otherwise

    Returns
    -------
    pixarea : float
      pixel area in square radian or square degree

    Notes
    -----
    Raise a ValueError exception if nside is not valid.

    Examples
    --------
    >>> import jax_healpy as hp
    >>> hp.nside2pixarea(128, degrees = True)  # doctest: +FLOAT_CMP
    0.2098234113027917

    >>> hp.nside2pixarea(256)
    1.5978966540475428e-05

    >>> hp.nside2pixarea(7)
    0.021371378595848933
    """

    pixarea = 4 * np.pi / nside2npix(nside)

    if degrees:
        pixarea = np.rad2deg(np.rad2deg(pixarea))

    return pixarea


def _lonlat2thetaphi(lon: ArrayLike, lat: ArrayLike):
    """Transform longitude and latitude (deg) into co-latitude and longitude (rad)

    Parameters
    ----------
    lon : int or array-like
      Longitude in degrees
    lat : int or array-like
      Latitude in degrees

    Returns
    -------
    theta, phi : float, scalar or array-like
      The co-latitude and longitude in radians
    """
    return np.pi / 2 - jnp.radians(lat), jnp.radians(lon)


def _thetaphi2lonlat(theta, phi):
    """Transform co-latitude and longitude (rad) into longitude and latitude (deg)

    Parameters
    ----------
    theta : int or array-like
      Co-latitude in radians
    phi : int or array-like
      Longitude in radians

    Returns
    -------
    lon, lat : float, scalar or array-like
      The longitude and latitude in degrees
    """
    return jnp.degrees(phi), 90.0 - jnp.degrees(theta)


def maptype(m):
    """Describe the type of the map (valid, single, sequence of maps).
    Checks : the number of maps, that all maps have same length and that this
    length is a valid map size (using :func:`isnpixok`).

    Parameters
    ----------
    m : sequence
      the map to get info from

    Returns
    -------
    info : int
      -1 if the given object is not a valid map, 0 if it is a single map,
      *info* > 0 if it is a sequence of maps (*info* is then the number of
      maps)

    Examples
    --------
    >>> import healpy as hp
    >>> hp.pixelfunc.maptype(np.arange(12))
    0
    >>> hp.pixelfunc.maptype([np.arange(12), np.arange(12)])
    2
    """
    if not hasattr(m, '__len__'):
        raise TypeError('input map is a scalar')
    if len(m) == 0:
        raise TypeError('input map has length zero')

    try:
        npix = len(m[0])
    except TypeError:
        npix = None

    if npix is not None:
        for mm in m[1:]:
            if len(mm) != npix:
                raise TypeError('input maps have different npix')
        if isnpixok(len(m[0])):
            return len(m)
        else:
            raise TypeError('bad number of pixels')
    else:
        if isnpixok(len(m)):
            return 0
        else:
            raise TypeError('bad number of pixels')


@partial(jit, static_argnames=['nside', 'nest', 'lonlat'])
def ang2pix(
    nside: int,
    theta: ArrayLike,
    phi: ArrayLike,
    nest: bool = False,
    lonlat: bool = False,
) -> Array:
    """ang2pix: nside,theta[rad],phi[rad],nest=False,lonlat=False -> ipix (default:RING)

    Unlike healpy.ang2pix, specifying a theta not in the range [0, π] does
    not raise an error, but returns -1.

    Parameters
    ----------
    nside : int, scalar or array-like
      The healpix nside parameter, must be a power of 2, less than 2**30
    theta, phi : float, scalars or array-like
      Angular coordinates of a point on the sphere
    nest : bool, optional
      if True, assume NESTED pixel ordering, otherwise, RING pixel ordering
    lonlat : bool
      If True, input angles are assumed to be longitude and latitude in degree,
      otherwise, they are co-latitude and longitude in radians.

    Returns
    -------
    pix : int or array of int
      The healpix pixel numbers. Scalar if all input are scalar, array otherwise.
      Usual numpy broadcasting rules apply.

    See Also
    --------
    pix2ang, pix2vec, vec2pix

    Examples
    --------
    Note that some of the test inputs below that are on pixel boundaries
    such as theta=π/2, phi=π/2, have a tiny value of 1e-15 added to them
    to make them reproducible on i386 machines using x87 floating point
    instruction set (see https://github.com/healpy/healpy/issues/528).

    >>> import jax_healpy as hp
    >>> from jax.numpy import pi as π
    >>> hp.ang2pix(16, π/2, 0)
    Array(1440, dtype=int64)

    >>> print(hp.ang2pix(16, np.array([π/2, π/4, π/2, 0, π]), np.array([0., π/4, π/2 + 1e-15, 0, 0])))
    [1440  427 1520    0 3068]

    >>> print(hp.ang2pix(16, π/2, np.array([0, π/2 + 1e-15])))
    [1440 1520]

    >>> print(hp.ang2pix(np.array([1, 2, 4, 8, 16]), π/2, 0))
    [   4   12   72  336 1440]

    >>> print(hp.ang2pix(np.array([1, 2, 4, 8, 16]), 0, 0, lonlat=True))
    [   4   12   72  336 1440]
    """
    check_nside(nside, nest=nest)

    if nest:
        raise NotImplementedError('NEST pixel ordering is not implemented.')

    if lonlat:
        theta, phi = _lonlat2thetaphi(theta, phi)

    pixels = _zphi2pix_ring(nside, jnp.cos(theta), jnp.sin(theta), phi)
    return jnp.where((theta < 0) | (theta > np.pi + 1e-5), -1, pixels)


def _zphi2pix_ring(nside: int, z: ArrayLike, sin_theta: ArrayLike, phi: ArrayLike) -> Array:
    tt = jnp.mod(2 * phi / np.pi, 4)
    ipix = jnp.where(
        jnp.abs(z) <= 2 / 3,
        _zphi2pix_equatorial_region_ring(nside, z, sin_theta, tt),
        _zphi2pix_polar_caps_ring(nside, z, sin_theta, tt),
    )
    return ipix


def _zphi2pix_equatorial_region_ring(nside: int, z: ArrayLike, sin_theta: float, tt: ArrayLike) -> Array:
    ncap = 2 * nside * (nside - 1)
    nl4 = 4 * nside
    jp = (nside * (0.5 + tt - 0.75 * z)).astype(int)
    jm = (nside * (0.5 + tt + 0.75 * z)).astype(int)
    ir = nside + 1 + jp - jm
    kshift = 1 - ir & 1  # ir even -> 1, odd -> 0
    t1 = jp + jm - nside + kshift + 1 + nl4 + nl4
    ip = (t1 >> 1) & (nl4 - 1)
    pix = ncap + (ir - 1) * nl4 + ip
    return pix


def _zphi2pix_polar_caps_ring(nside: int, z: ArrayLike, sin_theta: ArrayLike, tt: ArrayLike) -> Array:
    npixel = nside2npix(nside)
    tp = tt - jnp.floor(tt)
    #    tmp = nside * sin_theta / jnp.sqrt((1 + jnp.abs(z)) / 3)
    tmp = nside * jnp.sqrt(3.0 * (1.0 - jnp.abs(z)))
    jp = (tp * tmp).astype(int)
    jm = ((1.0 - tp) * tmp).astype(int)
    ir = jp + jm + 1
    ip = (tt * ir).astype(int)
    return jnp.where(z > 0, 2 * ir * (ir - 1) + ip, npixel - 2 * ir * (ir + 1) + ip)


@partial(jit, static_argnames=['nside', 'nest', 'lonlat'])
def pix2ang(nside: int, ipix: ArrayLike, nest: bool = False, lonlat: bool = False) -> tuple[Array, Array]:
    """pix2ang : nside,ipix,nest=False,lonlat=False -> theta[rad],phi[rad] (default RING)

    Parameters
    ----------
    nside : int or array-like
      The healpix nside parameter, must be a power of 2, less than 2**30
    ipix : int or array-like
      Pixel indices
    nest : bool, optional
      if True, assume NESTED pixel ordering, otherwise, RING pixel ordering
    lonlat : bool, optional
      If True, return angles will be longitude and latitude in degree,
      otherwise, angles will be co-latitude and longitude in radians (default)

    Returns
    -------
    theta, phi : float, scalar or array-like
      The angular coordinates corresponding to ipix. Scalar if all input
      are scalar, array otherwise. Usual numpy broadcasting rules apply.

    See Also
    --------
    ang2pix, vec2pix, pix2vec

    Examples
    --------
    >>> import jax_healpy as hp
    >>> hp.pix2ang(16, 1440)
    (1.5291175943723188, 0.0)

    >>> hp.pix2ang(16, [1440,  427, 1520,    0, 3068])
    (array([ 1.52911759,  0.78550497,  1.57079633,  0.05103658,  3.09055608]), array([ 0.        ,  0.78539816,  1.61988371,  0.78539816,  0.78539816]))

    >>> hp.pix2ang([1, 2, 4, 8], 11)
    (array([ 2.30052398,  0.84106867,  0.41113786,  0.2044802 ]), array([ 5.49778714,  5.89048623,  5.89048623,  5.89048623]))

    >>> hp.pix2ang([1, 2, 4, 8], 11, lonlat=True)
    (array([ 315. ,  337.5,  337.5,  337.5]), array([-41.8103149 ,  41.8103149 ,  66.44353569,  78.28414761]))
    """  # noqa: E501

    check_nside(nside, nest=nest)

    if nest:
        theta, phi = _pix2ang_nest(nside, ipix)
    else:
        iring = _pix2i_ring(nside, ipix)
        theta = _pix2theta_ring(nside, iring, ipix)
        phi = _pix2phi_ring(nside, iring, ipix)

    if lonlat:
        return _thetaphi2lonlat(theta, phi)
    return theta, phi


def _pix2i_ring(nside: int, pixels: ArrayLike) -> Array:
    npixel = nside2npix(nside)
    ncap = 2 * nside * (nside - 1)
    iring = jnp.where(
        pixels < ncap,
        _pix2i_north_cap_ring(nside, pixels),
        jnp.where(
            pixels < npixel - ncap,
            _pix2i_equatorial_region_ring(nside, pixels),
            _pix2i_south_cap_ring(nside, pixels),
        ),
    )
    return iring


def _pix2i_north_cap_ring(nside: int, pixels: ArrayLike) -> Array:
    return (1 + jnp.sqrt(1 + 2 * pixels).astype(int)) >> 1  # counted from North Pole


def _pix2i_equatorial_region_ring(nside: int, pixels: ArrayLike) -> Array:
    ncap = 2 * nside * (nside - 1)
    ip = pixels - ncap
    order = nside2order(nside)
    #   I tmp = (order_>=0) ? ip>>(order_+2) : ip/nl4;
    tmp = ip >> (order + 2)
    return tmp + nside


def _pix2i_south_cap_ring(nside: int, pixels: ArrayLike) -> Array:
    npixel = nside2npix(nside)
    ip = npixel - pixels
    return (1 + jnp.sqrt(2 * ip - 1).astype(int)) >> 1  # counted from South Pole


def _pix2z_ring(nside: int, iring: ArrayLike, pixels: ArrayLike) -> tuple[Array, Array]:
    npixel = nside2npix(nside)
    ncap = 2 * nside * (nside - 1)
    abs_one_minus_z = _pix2z_polar_caps_ring(nside, iring)
    z = jnp.where(
        pixels < ncap,
        1 - abs_one_minus_z,
        jnp.where(
            pixels < npixel - ncap,
            _pix2z_equatorial_region_ring(nside, iring),
            abs_one_minus_z - 1,
        ),
    )
    return z, abs_one_minus_z


def _pix2z_polar_caps_ring(nside: int, iring: ArrayLike) -> Array:
    npixel = nside2npix(nside)
    return iring * iring * 4 / npixel


def _pix2z_equatorial_region_ring(nside: int, iring: ArrayLike) -> Array:
    return (2 * nside - iring) * 2 / 3 / nside


def _pix2theta_ring(nside: int, iring: ArrayLike, pixels: ArrayLike) -> Array:
    z, abs_one_minus_z = _pix2z_ring(nside, iring, pixels)
    theta = jnp.where(
        jnp.abs(z) > 0.99,
        jnp.arctan2(jnp.sqrt(abs_one_minus_z * (2 - abs_one_minus_z)), z),
        jnp.arccos(z),
    )

    return theta


def _pix2phi_ring(nside: int, iring: ArrayLike, pixels: ArrayLike) -> Array:
    npixel = nside2npix(nside)
    ncap = 2 * nside * (nside - 1)
    phi = jnp.where(
        pixels < ncap,
        _pix2phi_north_cap_ring(nside, iring, pixels),
        jnp.where(
            pixels < npixel - ncap,
            _pix2phi_equatorial_region_ring(nside, iring, pixels),
            _pix2phi_south_cap_ring(nside, iring, pixels),
        ),
    )
    return phi


def _pix2phi_north_cap_ring(nside: int, iring: ArrayLike, pixels: ArrayLike) -> Array:
    iphi = pixels + 1 - 2 * iring * (iring - 1)
    phi = (iphi - 0.5) * np.pi / 2 / iring
    return phi


def _pix2phi_equatorial_region_ring(nside: int, iring: ArrayLike, pixels: ArrayLike) -> Array:
    iphi = pixels + 2 * nside * (nside + 1) - 4 * nside * iring + 1
    fodd = ((iring + nside) & 1) * 0.5 + 0.5  # iring + nside odd -> 1 else 0.5
    phi = (iphi - fodd) * np.pi / 2 / nside
    return phi


def _pix2phi_south_cap_ring(nside: int, iring: ArrayLike, pixels: ArrayLike) -> Array:
    npixel = nside2npix(nside)
    iphi = 4 * iring + 1 - (npixel - pixels - 2 * iring * (iring - 1))
    phi = (iphi - 0.5) * np.pi / 2 / iring
    return phi


def _pix2ang_nest(nside: ArrayLike, ipix: ArrayLike) -> tuple[Array, Array]:
    raise NotImplementedError('NEST pixel ordering is not implemented.')


# template<typename I> void T_Healpix_Base<I>::pix2loc (I pix, double &z,
#   double &phi, double &sth, bool &have_sth) const
#   have_sth=false;
#   {
#   int face_num, ix, iy;
#   nest2xyf(pix,ix,iy,face_num);
#
#   I jr = (I(jrll[face_num])<<order_) - ix - iy - 1;
#
#   I nr;
#   if (jr<nside_)
#     {
#     nr = jr;
#     double tmp=(nr*nr)*fact2_;
#     z = 1 - tmp;
#     if (z>0.99) { sth=sqrt(tmp*(2.-tmp)); have_sth=true; }
#     }
#   else if (jr > 3*nside_)
#     {
#     nr = nside_*4-jr;
#     double tmp=(nr*nr)*fact2_;
#     z = tmp - 1;
#     if (z<-0.99) { sth=sqrt(tmp*(2.-tmp)); have_sth=true; }
#     }
#   else
#     {
#     nr = nside_;
#     z = (2*nside_-jr)*fact1_;
#     }
#
#   I tmp=I(jpll[face_num])*nr+ix-iy;
#   planck_assert(tmp<8*nr,"must not happen");
#   if (tmp<0) tmp+=8*nr;
#   phi = (nr==nside_) ? 0.75*halfpi*tmp*fact1_ :
#                        (0.5*halfpi*tmp)/nr;
#   }
# }


@partial(jit, static_argnames=['nside', 'nest'])
def vec2pix(nside: int, x: ArrayLike, y: ArrayLike, z: ArrayLike, nest: bool = False) -> Array:
    """vec2pix : nside,x,y,z,nest=False -> ipix (default:RING)

    Parameters
    ----------
    nside : int or array-like
      The healpix nside parameter, must be a power of 2, less than 2**30
    x,y,z : floats or array-like
      vector coordinates defining point on the sphere
    nest : bool, optional
      if True, assume NESTED pixel ordering, otherwise, RING pixel ordering

    Returns
    -------
    ipix : int, scalar or array-like
      The healpix pixel number corresponding to input vector. Scalar if all input
      are scalar, array otherwise. Usual numpy broadcasting rules apply.

    See Also
    --------
    ang2pix, pix2ang, pix2vec

    Examples
    --------
    >>> import healpy as hp
    >>> hp.vec2pix(16, 1, 0, 0)
    1504

    >>> print(hp.vec2pix(16, [1, 0], [0, 1], [0, 0]))
    [1504 1520]

    >>> print(hp.vec2pix([1, 2, 4, 8], 1, 0, 0))
    [  4  20  88 368]
    """
    check_nside(nside, nest=nest)
    if nest:
        raise NotImplementedError

    return _vec2pix_ring(nside, x, y, z)


def vec2pix2(nside: int, vec: ArrayLike, nest: bool = False) -> Array:
    return vec2pix2_ring(nside, vec)


@partial(jit, static_argnames='nside')
@partial(vmap, in_axes=(None, 1))
def vec2pix2_ring(nside: int, vec: ArrayLike) -> Array:
    vec /= jnp.sqrt(jnp.sum(vec**2))
    phi = jnp.arctan2(vec[1], vec[0])
    # return _zphi2pix_ring(nside, vec[2], jnp.sqrt(vec[0] ** 2 + vec[1] ** 2), phi)
    return _zphi2pix_ring(nside, vec[2], jnp.sqrt(vec[0] ** 2 + vec[1] ** 2), phi)


def _vec2pix_ring(nside: int, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> Array:
    dnorm = 1 / jnp.sqrt(x**2 + y**2 + z**2)
    z *= dnorm
    phi = jnp.arctan2(y, x)
    return _zphi2pix_ring(nside, z, jnp.sqrt(x**2 + y**2) * dnorm, phi)


@partial(jit, static_argnames=['nside', 'nest'])
def pix2vec(nside: int, ipix: ArrayLike, nest: bool = False) -> Array:
    """pix2vec : nside,ipix,nest=False -> x,y,z (default RING)

    Parameters
    ----------
    nside : int, scalar or array-like
      The healpix nside parameter, must be a power of 2, less than 2**30
    ipix : int, scalar or array-like
      Healpix pixel number
    nest : bool, optional
      if True, assume NESTED pixel ordering, otherwise, RING pixel ordering

    Returns
    -------
    x, y, z : floats, scalar or array-like
      The coordinates of vector corresponding to input pixels. Scalar if all input
      are scalar, array otherwise. Usual numpy broadcasting rules apply.

    See Also
    --------
    ang2pix, pix2ang, vec2pix

    Examples
    --------
    >>> import healpy as hp
    >>> hp.pix2vec(16, 1504)
    (0.99879545620517241, 0.049067674327418015, 0.0)

    >>> hp.pix2vec(16, [1440,  427])
    (array([ 0.99913157,  0.5000534 ]), array([ 0.       ,  0.5000534]), array([ 0.04166667,  0.70703125]))

    >>> hp.pix2vec([1, 2], 11)
    (array([ 0.52704628,  0.68861915]), array([-0.52704628, -0.28523539]), array([-0.66666667,  0.66666667]))
    """
    check_nside(nside, nest=nest)
    if nest:
        raise NotImplementedError

    return _pix2vec_ring(nside, ipix)


def _pix2vec_ring(nside, pixels):
    iring = _pix2i_ring(nside, pixels)
    z, abs_one_minus_z = _pix2z_ring(nside, iring, pixels)
    phi = _pix2phi_ring(nside, iring, pixels)
    sin_theta = jnp.sqrt(
        jnp.where(
            jnp.abs(z) > 0.99,
            abs_one_minus_z * (2 - abs_one_minus_z),
            (1 - z) * (1 + z),
        )
    )
    return jnp.array([sin_theta * jnp.cos(phi), sin_theta * jnp.sin(phi), z]).T


@partial(jit, static_argnames=['lonlat'])
def ang2vec(theta: ArrayLike, phi: ArrayLike, lonlat: bool = False) -> Array:
    """ang2vec : convert angles to 3D position vector

    Parameters
    ----------
    theta : float, scalar or array-like
      co-latitude in radians measured southward from the North pole (in [0,pi]).
    phi : float, scalar or array-like
      longitude in radians measured eastward (in [0, 2*pi]).
    lonlat : bool
      If True, input angles are assumed to be longitude and latitude in degree,
      otherwise, they are co-latitude and longitude in radians.

    Returns
    -------
    vec : float, array
      if theta and phi are vectors, the result is a 2D array with a vector per row
      otherwise, it is a 1D array of shape (3,)

    See Also
    --------
    vec2ang, rotator.dir2vec, rotator.vec2dir
    """
    if lonlat:
        theta, phi = _lonlat2thetaphi(theta, phi)

    theta = jnp.where((theta < 0) | (theta > np.pi + 1e-5), np.nan, theta)
    sin_theta = jnp.sin(theta)
    x = sin_theta * jnp.cos(phi)
    y = sin_theta * jnp.sin(phi)
    z = jnp.cos(theta)
    return jnp.array([x, y, z]).T


@partial(jit, static_argnames=['lonlat'])
def vec2ang(vectors: ArrayLike, lonlat: bool = False) -> tuple[Array, Array]:
    """vec2ang: vectors [x, y, z] -> theta[rad], phi[rad]

    Parameters
    ----------
    vectors : float, array-like
      the vector(s) to convert, shape is (3,) or (N, 3)
    lonlat : bool, optional
      If True, return angles will be longitude and latitude in degree,
      otherwise, angles will be co-latitude and longitude in radians (default)

    Returns
    -------
    theta, phi : float, tuple of two arrays
      the colatitude and longitude in radians

    See Also
    --------
    ang2vec, rotator.vec2dir, rotator.dir2vec
    """
    vectors = vectors.reshape(-1, 3)
    dnorm = jnp.sqrt(vectors[..., 0] ** 2 + vectors[..., 1] ** 2 + vectors[..., 2] ** 2)
    theta = jnp.arccos(vectors[:, 2] / dnorm)
    phi = jnp.arctan2(vectors[:, 1], vectors[:, 0])
    phi = jnp.where(phi < 0, phi + 2 * np.pi, phi)
    if lonlat:
        return _thetaphi2lonlat(theta, phi)
    return theta, phi
