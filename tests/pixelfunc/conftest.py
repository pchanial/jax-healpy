import numpy as np
import pytest
from jaxtyping import Array


@pytest.fixture
def theta0() -> Array:
    return np.array([1.52911759, 0.78550497, 1.57079633, 0.05103658, 3.09055608])


@pytest.fixture
def phi0() -> Array:
    return np.array([0.0, 0.78539816, 1.61988371, 0.78539816, 0.78539816])


@pytest.fixture
def lon0(phi0: float) -> Array:
    return np.degrees(phi0)


@pytest.fixture
def lat0(theta0: float) -> Array:
    return 90.0 - np.degrees(theta0)


@pytest.fixture
def vec0() -> np.ndarray:
    return np.array(
        [
            [9.99131567e-01, 0.00000000e00, 4.16666710e-02],
            [5.00053402e-01, 5.00053399e-01, 7.07031253e-01],
            [-4.90676723e-02, 9.98795456e-01, -3.20510345e-09],
            [3.60726472e-02, 3.60726470e-02, 9.98697916e-01],
            [3.60726427e-02, 3.60726425e-02, -9.98697917e-01],
        ]
    )
