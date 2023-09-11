import numpy as np
import pytest
from numpy.testing import assert_allclose

import jax_healpy as hp


@pytest.mark.parametrize(
    'nest',
    [False, pytest.param(True, marks=pytest.mark.xfail(reason='NEST not implemented'))],
)
def test_ang2pix2ang(theta0: float, phi0: float, nest: bool) -> None:
    # ensure nside = 1 << 23 is correctly calculated
    # by comparing the original theta phi are restored.
    # NOTE: nside needs to be sufficiently large, otherwise the angles associated to the pixel center will differ
    # from the input angles which may be off-center.
    nside = hp.order2nside(23)
    pixels = hp.ang2pix(nside, theta0, phi0, nest=nest)
    actual_theta, actual_phi = hp.pix2ang(nside, pixels, nest=nest)
    assert np.allclose(actual_theta, theta0)
    assert np.allclose(actual_phi, phi0)


@pytest.mark.parametrize(
    'nest',
    [False, pytest.param(True, marks=pytest.mark.xfail(reason='NEST not implemented'))],
)
def test_ang2pix2ang_lonlat(lon0: float, lat0: float, nest: bool) -> None:
    # Need to decrease the precision of the check because deg not radians
    nside = hp.order2nside(23)
    pixels = hp.ang2pix(nside, lon0, lat0, nest=nest, lonlat=True)
    actual_lon, actual_lat = hp.pix2ang(nside, pixels, nest=nest, lonlat=True)
    np.testing.assert_array_almost_equal(actual_lon, lon0, decimal=5)
    np.testing.assert_array_almost_equal(actual_lat, lat0, decimal=5)


@pytest.mark.parametrize(
    'nside, z, iphi, expected_pixel',
    [
        (2, 0.999999999999, 0, [0, 0, 1, 2, 3]),
        (2, 0.999, 0, [0, 0, 1, 2, 3]),
        (2, 0.98, 0, [0, 0, 1, 2, 3]),
        (2, 0.2, 0, [12, 21, 15, 17, 26]),
        (2, 0, 0, [12, 21, 23, 24, 26]),
        (2, -0.2, 0, [28, 21, 31, 33, 26]),
        (2, -0.98, 0, [44, 44, 45, 46, 47]),
        (2, -0.999, 0, [44, 44, 45, 46, 47]),
        (2, -0.999999999999, 0, [44, 44, 45, 46, 47]),
        (256, 0.999999999999, 0, [0, 0, 1, 2, 3]),
        (256, 0.999, 0, [420, 375, 386, 397, 408]),
        (256, 0.98, 0, [7812, 7862, 7912, 7963, 8013]),
        (256, 0.2, 0, [313856, 314061, 314266, 315494, 314675]),
        (256, 0, 0, [391680, 392908, 393113, 393318, 393523]),
        (256, -0.2, 0, [471552, 471757, 471962, 471142, 472371]),
        (256, -0.98, 0, [778368, 778418, 778468, 778519, 778569]),
        (256, -0.999, 0, [785952, 786023, 786034, 786045, 786056]),
        (256, -0.999999999999, 0, [786428, 786428, 786429, 786430, 786431]),
        (8388608, 0.999999999999, 0, [420, 375, 386, 397, 408]),
        (8388608, 0.999, 0, [422211577812, 422211945382, 422212312952, 422212680523, 422213048093]),
        (8388608, 0.98, 0, [8444245806360, 8444247450184, 8444249094009, 8444250737834, 8444252381659]),
        (8388608, 0.2, 0, [337769935142912, 337769975408230, 337769948564685, 337769988830003, 337769995540889]),
        (8388608, 0, 0, [422212414734336, 422212454999654, 422212461710540, 422212468421427, 422212475132313]),
        (8388608, -0.2, 0, [506654961434624, 506654934591078, 506654974856397, 506654948012851, 506654954723737]),
        (8388608, -0.98, 0, [835980676106484, 835980677750308, 835980679394133, 835980681037958, 835980682681783]),
        (8388608, -0.999, 0, [844002716716304, 844002717083874, 844002717451444, 844002717819015, 844002718186585]),
        (
            8388608,
            -0.999999999999,
            0,
            [844424930131488, 844424930131559, 844424930131570, 844424930131581, 844424930131592],
        ),
    ],
)
def test_ang2pix_ring(nside: int, z: float, iphi: int, expected_pixel: int) -> None:
    # test that we get the same results as healpy
    theta = np.arccos(z)
    phi = np.arange(5) / 5 * 2 * np.pi
    actual_pixel = hp.ang2pix(nside, theta, phi)
    assert list(actual_pixel) == expected_pixel


@pytest.mark.parametrize(
    'nside, pixel, expected_theta, expected_phi',
    [
        (2, 0, 0.4111378623223478, 0.7853981633974483),
        (2, 1, 0.4111378623223478, 2.356194490192345),
        (2, 2, 0.4111378623223478, 3.9269908169872414),
        (2, 3, 0.4111378623223478, 5.497787143782138),
        (2, 12, 1.2309594173407747, 0.0),
        (2, 15, 1.2309594173407747, 2.356194490192345),
        (2, 17, 1.2309594173407747, 3.926990816987241),
        (2, 21, 1.5707963267948966, 1.1780972450961724),
        (2, 23, 1.5707963267948966, 2.748893571891069),
        (2, 24, 1.5707963267948966, 3.534291735288517),
        (2, 26, 1.5707963267948966, 5.105088062083414),
        (2, 28, 1.9106332362490186, 0.0),
        (2, 31, 1.9106332362490186, 2.356194490192345),
        (2, 33, 1.9106332362490186, 3.926990816987241),
        (2, 44, 2.7304547912674453, 0.7853981633974483),
        (2, 45, 2.7304547912674453, 2.356194490192345),
        (2, 46, 2.7304547912674453, 3.9269908169872414),
        (2, 47, 2.7304547912674453, 5.497787143782138),
        (256, 0, 0.0031894411211112732, 0.7853981633974483),
        (256, 1, 0.0031894411211112732, 2.356194490192345),
        (256, 2, 0.0031894411211112732, 3.9269908169872414),
        (256, 3, 0.0031894411211112732, 5.497787143782138),
        (256, 375, 0.04465586710781478, 1.2902969827243793),
        (256, 386, 0.04465586710781478, 2.524494096634655),
        (256, 397, 0.04465586710781478, 3.758691210544931),
        (256, 408, 0.04465586710781478, 4.992888324455207),
        (256, 420, 0.04784616024413705, 0.05235987755982988),
        (256, 7812, 0.20127427886846755, 0.012466637514245212),
        (256, 7862, 0.20127427886846755, 1.2591303889387664),
        (256, 7912, 0.20127427886846755, 2.5057941403632875),
        (256, 7963, 0.20127427886846755, 3.7773911668162987),
        (256, 8013, 0.20127427886846755, 5.02405491824082),
        (256, 313856, 1.3689068038418237, 0.0),
        (256, 314061, 1.3689068038418237, 1.2578642460662257),
        (256, 314266, 1.3689068038418237, 2.5157284921324514),
        (256, 314675, 1.3689068038418237, 5.0253210611133605),
        (256, 315494, 1.3715642395497258, 3.7705247766229055),
        (256, 391680, 1.5681921571847817, 0.0),
        (256, 392908, 1.5707963267948966, 1.2547962844904545),
        (256, 393113, 1.5707963267948966, 2.5126605305566803),
        (256, 393318, 1.5707963267948966, 3.7705247766229055),
        (256, 393523, 1.5707963267948966, 5.028389022689131),
        (256, 471142, 1.7700284140400675, 3.7705247766229055),
        (256, 471552, 1.7726858497479696, 0.0),
        (256, 471757, 1.7726858497479696, 1.2578642460662257),
        (256, 471962, 1.7726858497479696, 2.5157284921324514),
        (256, 472371, 1.7726858497479696, 5.0253210611133605),
        (256, 778368, 2.9403183747213255, 0.012466637514245212),
        (256, 778418, 2.9403183747213255, 1.2591303889387664),
        (256, 778468, 2.9403183747213255, 2.5057941403632875),
        (256, 778519, 2.9403183747213255, 3.7773911668162987),
        (256, 778569, 2.9403183747213255, 5.02405491824082),
        (256, 785952, 3.093746493345656, 0.05235987755982988),
        (256, 786023, 3.0969367864819786, 1.2902969827243793),
        (256, 786034, 3.0969367864819786, 2.524494096634655),
        (256, 786045, 3.0969367864819786, 3.758691210544931),
        (256, 786056, 3.0969367864819786, 4.992888324455207),
        (256, 786428, 3.138403212468682, 0.7853981633974483),
        (256, 786429, 3.138403212468682, 2.356194490192345),
        (256, 786430, 3.138403212468682, 3.9269908169872414),
        (256, 786431, 3.138403212468682, 5.497787143782138),
        (8388608, 375, 1.3626756826626123e-06, 1.2902969827243793),
        (8388608, 386, 1.3626756826626123e-06, 2.524494096634655),
        (8388608, 397, 1.3626756826626123e-06, 3.758691210544931),
        (8388608, 408, 1.3626756826626123e-06, 4.992888324455207),
        (8388608, 420, 1.4600096599956724e-06, 0.05235987755982988),
        (8388608, 422211577812, 0.04472508884652595, 1.7093828303855769e-06),
        (8388608, 422211945382, 0.04472508884652595, 1.2566374033124834),
        (8388608, 422212312952, 0.04472508884652595, 2.513273097242136),
        (8388608, 422212680523, 0.04472508884652595, 3.7699122099374502),
        (8388608, 422213048093, 0.04472508884652595, 5.026547903867104),
        (8388608, 8444245806360, 0.20033484963814002, 3.8222962125766605e-07),
        (8388608, 8444247450184, 0.20033484963814002, 1.2566368320981445),
        (8388608, 8444249094009, 0.20033484963814002, 2.5132740464259102),
        (8388608, 8444250737834, 0.20033484963814002, 3.769911260753676),
        (8388608, 8444252381659, 0.20033484963814002, 5.026548475081441),
        (8388608, 337769935142912, 1.369438357337577, 0.0),
        (8388608, 337769948564685, 1.369438357337577, 2.5132741603225375),
        (8388608, 337769975408230, 1.3694384384492249, 1.2566370801612687),
        (8388608, 337769988830003, 1.3694384384492249, 3.7699112404838058),
        (8388608, 337769995540889, 1.3694384384492249, 5.026548227018318),
        (8388608, 422212414734336, 1.570796247322037, 0.0),
        (8388608, 422212454999654, 1.5707963267948966, 1.2566370801612687),
        (8388608, 422212461710540, 1.5707963267948966, 2.5132740666957805),
        (8388608, 422212468421427, 1.5707963267948966, 3.7699112404838058),
        (8388608, 422212475132313, 1.5707963267948966, 5.026548227018318),
        (8388608, 506654934591078, 1.7721542151405685, 1.2566370801612687),
        (8388608, 506654948012851, 1.7721542151405685, 3.7699112404838058),
        (8388608, 506654954723737, 1.7721542151405685, 5.026548227018318),
        (8388608, 506654961434624, 1.7721542962522163, 0.0),
        (8388608, 506654974856397, 1.7721542962522163, 2.5132741603225375),
        (8388608, 835980676106484, 2.941257803951653, 3.8222962125766605e-07),
        (8388608, 835980677750308, 2.941257803951653, 1.2566368320981445),
        (8388608, 835980679394133, 2.941257803951653, 2.5132740464259102),
        (8388608, 835980681037958, 2.941257803951653, 3.769911260753676),
        (8388608, 835980682681783, 2.941257803951653, 5.026548475081441),
        (8388608, 844002716716304, 3.0968675647432673, 1.7093828303855769e-06),
        (8388608, 844002717083874, 3.0968675647432673, 1.2566374033124834),
        (8388608, 844002717451444, 3.0968675647432673, 2.513273097242136),
        (8388608, 844002717819015, 3.0968675647432673, 3.7699122099374502),
        (8388608, 844002718186585, 3.0968675647432673, 5.026547903867104),
        (8388608, 844424930131488, 3.1415911935801333, 0.05235987755982988),
        (8388608, 844424930131559, 3.1415912909141106, 1.2902969827243793),
        (8388608, 844424930131570, 3.1415912909141106, 2.524494096634655),
        (8388608, 844424930131581, 3.1415912909141106, 3.758691210544931),
        (8388608, 844424930131592, 3.1415912909141106, 4.992888324455207),
    ],
)
def test_pix2ang_ring(nside: int, pixel: int, expected_theta: float, expected_phi: float) -> None:
    # test that we get the same results as healpy
    actual_theta, actual_phi = hp.pix2ang(nside, pixel)
    assert_allclose(actual_theta, expected_theta, rtol=1e-12, atol=1e-14)
    assert_allclose(actual_phi, expected_phi, rtol=1e-12, atol=1e-14)


@pytest.mark.skip(reason='NEST ordering not implemented')
@pytest.mark.parametrize(
    'nside, z, expected_pixel',
    [
        (2, 0.999999999999, [3, 3, 7, 11, 15]),
        (2, 0.999, [3, 3, 7, 11, 15]),
        (2, 0.98, [3, 3, 7, 11, 15]),
        (2, 0.2, [19, 22, 4, 8, 29]),
        (2, 0, [19, 22, 26, 25, 29]),
        (2, -0.2, [16, 22, 39, 43, 29]),
        (2, -0.98, [32, 32, 36, 40, 44]),
        (2, -0.999, [32, 32, 36, 40, 44]),
        (2, -0.999999999999, [32, 32, 36, 40, 44]),
        (256, 0.999999999999, [65535, 65535, 131071, 196607, 262143]),
        (256, 0.999, [65451, 65393, 130926, 196509, 262066]),
        (256, 0.98, [64171, 62887, 128668, 194924, 260699]),
        (256, 0.2, [314428, 374663, 69792, 133882, 490315]),
        (256, 0, [311296, 367194, 436585, 415382, 484773]),
        (256, -0.2, [275395, 361652, 647087, 719370, 477304]),
        (256, -0.98, [527016, 525732, 591507, 657763, 723544]),
        (256, -0.999, [524456, 524365, 589922, 655505, 721038]),
        (256, -0.999999999999, [524288, 524288, 589824, 655360, 720896]),
        (8388608, 0.999999999999, [70368744177579, 70368744177521, 140737488355182, 211106232532893, 281474976710578]),
        (8388608, 0.999, [70278549581803, 70215893321959, 140581352114460, 211000373347884, 281391603516635]),
        (8388608, 0.98, [68903870655151, 67525170488751, 138156734224031, 209298842069359, 279923936516703]),
        (8388608, 0.2, [337614746881807, 402291901456865, 74939263100968, 143755755568830, 526472038240978]),
        (8388608, 0, [334251534843904, 394271934289558, 468780016360026, 446013657949605, 520521740020073]),
        (8388608, -0.2, [295703950717168, 388321636068653, 694805112548331, 772417698038402, 512501772852766]),
        (8388608, -0.98, [565879700466336, 564500993615264, 635126088062608, 706268195907936, 776899759643216]),
        (8388608, -0.999, [563130342613032, 563033326615332, 633424556784083, 703843578017507, 774209036810008]),
        (
            8388608,
            -0.999999999999,
            [562949953421480, 562949953421389, 633318697599074, 703687441776785, 774056185954446],
        ),
    ],
)
def test_ang2pix_nest(nside: int, z: float, expected_pixel: int) -> None:
    # test that we get the same results as healpy
    theta = np.arccos(z)
    phi = np.arange(5) / 5 * 2 * np.pi
    actual_pixel = hp.ang2pix(nside, theta, phi, nest=True)
    assert list(actual_pixel) == expected_pixel


@pytest.mark.skip(reason='NEST ordering not implemented')
@pytest.mark.parametrize(
    'nside, pixel, expected_theta, expected_phi',
    [
        (2, 3, 0.4111378623223478, 0.7853981633974483),
        (2, 4, 1.2309594173407747, 2.356194490192345),
        (2, 7, 0.4111378623223478, 2.356194490192345),
        (2, 8, 1.2309594173407747, 3.926990816987241),
        (2, 11, 0.4111378623223478, 3.9269908169872414),
        (2, 15, 0.4111378623223478, 5.497787143782138),
        (2, 16, 1.9106332362490186, 0.0),
        (2, 19, 1.2309594173407747, 0.0),
        (2, 22, 1.5707963267948966, 1.1780972450961724),
        (2, 25, 1.5707963267948966, 3.534291735288517),
        (2, 26, 1.5707963267948966, 2.748893571891069),
        (2, 29, 1.5707963267948966, 5.105088062083413),
        (2, 32, 2.7304547912674453, 0.7853981633974483),
        (2, 36, 2.7304547912674453, 2.356194490192345),
        (2, 39, 1.9106332362490186, 2.356194490192345),
        (2, 40, 2.7304547912674453, 3.9269908169872414),
        (2, 43, 1.9106332362490186, 3.926990816987241),
        (2, 44, 2.7304547912674453, 5.497787143782138),
        (256, 62887, 0.20127427886846755, 1.2591303889387664),
        (256, 64171, 0.20127427886846755, 0.012466637514245212),
        (256, 65393, 0.04465586710781478, 1.2902969827243793),
        (256, 65451, 0.04784616024413705, 0.05235987755982988),
        (256, 65535, 0.0031894411211112732, 0.7853981633974483),
        (256, 69792, 1.3689068038418237, 2.5157284921324514),
        (256, 128668, 0.20127427886846755, 2.5057941403632875),
        (256, 130926, 0.04465586710781478, 2.524494096634655),
        (256, 131071, 0.0031894411211112732, 2.356194490192345),
        (256, 133882, 1.3715642395497258, 3.7705247766229055),
        (256, 194924, 0.20127427886846755, 3.7773911668162987),
        (256, 196509, 0.04465586710781478, 3.758691210544931),
        (256, 196607, 0.0031894411211112732, 3.9269908169872414),
        (256, 260699, 0.20127427886846755, 5.02405491824082),
        (256, 262066, 0.04465586710781478, 4.992888324455207),
        (256, 262143, 0.0031894411211112732, 5.497787143782138),
        (256, 275395, 1.7726858497479696, 0.0),
        (256, 311296, 1.5681921571847817, 0.0),
        (256, 314428, 1.3689068038418237, 0.0),
        (256, 361652, 1.7726858497479696, 1.2578642460662257),
        (256, 367194, 1.5707963267948966, 1.2547962844904545),
        (256, 374663, 1.3689068038418237, 1.2578642460662257),
        (256, 415382, 1.5707963267948966, 3.7705247766229055),
        (256, 436585, 1.5707963267948966, 2.51266053055668),
        (256, 477304, 1.7726858497479696, 5.02532106111336),
        (256, 484773, 1.5707963267948966, 5.028389022689131),
        (256, 490315, 1.3689068038418237, 5.02532106111336),
        (256, 524288, 3.138403212468682, 0.7853981633974483),
        (256, 524365, 3.0969367864819786, 1.2902969827243793),
        (256, 524456, 3.093746493345656, 0.05235987755982988),
        (256, 525732, 2.9403183747213255, 1.2591303889387664),
        (256, 527016, 2.9403183747213255, 0.012466637514245212),
        (256, 589824, 3.138403212468682, 2.356194490192345),
        (256, 589922, 3.0969367864819786, 2.524494096634655),
        (256, 591507, 2.9403183747213255, 2.5057941403632875),
        (256, 647087, 1.7726858497479696, 2.5157284921324514),
        (256, 655360, 3.138403212468682, 3.9269908169872414),
        (256, 655505, 3.0969367864819786, 3.758691210544931),
        (256, 657763, 2.9403183747213255, 3.7773911668162987),
        (256, 719370, 1.7700284140400675, 3.7705247766229055),
        (256, 720896, 3.138403212468682, 5.497787143782138),
        (256, 721038, 3.0969367864819786, 4.992888324455207),
        (256, 723544, 2.9403183747213255, 5.02405491824082),
        (8388608, 67525170488751, 0.20033484963814002, 1.2566368320981445),
        (8388608, 68903870655151, 0.20033484963814002, 3.8222962125766605e-07),
        (8388608, 70215893321959, 0.04472508884652595, 1.2566374033124834),
        (8388608, 70278549581803, 0.04472508884652595, 1.7093828303855769e-06),
        (8388608, 70368744177521, 1.3626756826626123e-06, 1.2902969827243793),
        (8388608, 70368744177579, 1.4600096599956724e-06, 0.05235987755982988),
        (8388608, 74939263100968, 1.369438357337577, 2.513274160322537),
        (8388608, 138156734224031, 0.20033484963814002, 2.5132740464259102),
        (8388608, 140581352114460, 0.04472508884652595, 2.513273097242136),
        (8388608, 140737488355182, 1.3626756826626123e-06, 2.524494096634655),
        (8388608, 143755755568830, 1.3694384384492249, 3.7699112404838058),
        (8388608, 209298842069359, 0.20033484963814002, 3.769911260753676),
        (8388608, 211000373347884, 0.04472508884652595, 3.7699122099374502),
        (8388608, 211106232532893, 1.3626756826626123e-06, 3.758691210544931),
        (8388608, 279923936516703, 0.20033484963814002, 5.026548475081441),
        (8388608, 281391603516635, 0.04472508884652595, 5.026547903867104),
        (8388608, 281474976710578, 1.3626756826626123e-06, 4.992888324455207),
        (8388608, 295703950717168, 1.7721542962522163, 0.0),
        (8388608, 334251534843904, 1.570796247322037, 0.0),
        (8388608, 337614746881807, 1.369438357337577, 0.0),
        (8388608, 388321636068653, 1.7721542151405685, 1.2566370801612685),
        (8388608, 394271934289558, 1.5707963267948966, 1.2566370801612685),
        (8388608, 402291901456865, 1.3694384384492249, 1.2566370801612685),
        (8388608, 446013657949605, 1.5707963267948966, 3.7699112404838058),
        (8388608, 468780016360026, 1.5707963267948966, 2.51327406669578),
        (8388608, 512501772852766, 1.7721542151405685, 5.026548227018317),
        (8388608, 520521740020073, 1.5707963267948966, 5.026548227018317),
        (8388608, 526472038240978, 1.3694384384492249, 5.026548227018317),
        (8388608, 562949953421389, 3.1415912909141106, 1.2902969827243793),
        (8388608, 562949953421480, 3.1415911935801333, 0.05235987755982988),
        (8388608, 563033326615332, 3.0968675647432673, 1.2566374033124834),
        (8388608, 563130342613032, 3.0968675647432673, 1.7093828303855769e-06),
        (8388608, 564500993615264, 2.941257803951653, 1.2566368320981445),
        (8388608, 565879700466336, 2.941257803951653, 3.8222962125766605e-07),
        (8388608, 633318697599074, 3.1415912909141106, 2.524494096634655),
        (8388608, 633424556784083, 3.0968675647432673, 2.513273097242136),
        (8388608, 635126088062608, 2.941257803951653, 2.5132740464259102),
        (8388608, 694805112548331, 1.7721542962522163, 2.513274160322537),
        (8388608, 703687441776785, 3.1415912909141106, 3.758691210544931),
        (8388608, 703843578017507, 3.0968675647432673, 3.7699122099374502),
        (8388608, 706268195907936, 2.941257803951653, 3.769911260753676),
        (8388608, 772417698038402, 1.7721542151405685, 3.7699112404838058),
        (8388608, 774056185954446, 3.1415912909141106, 4.992888324455207),
        (8388608, 774209036810008, 3.0968675647432673, 5.026547903867104),
        (8388608, 776899759643216, 2.941257803951653, 5.026548475081441),
    ],
)
def test_pix2ang_nest(nside: int, pixel: int, expected_theta: float, expected_phi: float) -> None:
    # test that we get the same results as healpy
    actual_theta, actual_phi = hp.pix2ang(nside, pixel, nest=True)
    assert_allclose(actual_theta, expected_theta, rtol=1e-14)
    assert_allclose(actual_phi, expected_phi, rtol=1e-14)


def test_ang2pix_ring_outofrange(theta0, phi0):
    # Healpy_Base2 works up to nside = 2**29.
    # Check that a ValueError is raised for nside = 2**30.
    with pytest.raises(ValueError):
        hp.ang2pix(1 << 30, theta0, phi0, nest=False)


@pytest.mark.parametrize('nest', [False, True])
def test_ang2pix_nest_outofrange_doesntcrash(theta0: float, phi0: float, nest: bool) -> None:
    # Healpy_Base2 works up to nside = 2**29.
    # Check that a ValueError is raised for nside = 2**30.
    with pytest.raises(ValueError):
        hp.ang2pix(1 << 30, theta0, phi0, nest=nest)


@pytest.mark.parametrize(
    'nest',
    [False, pytest.param(True, marks=pytest.mark.xfail(reason='NEST not implemented'))],
)
@pytest.mark.parametrize('theta', [-1, np.pi + 1e-4])
def test_ang2pix_outofrange_theta(nest: bool, theta: float) -> None:
    pixel = hp.ang2pix(32, theta, 0, nest=nest)
    assert pixel == -1