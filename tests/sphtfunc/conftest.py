from typing import Any, Callable

import numpy as np
import pytest
from s2fft.sampling.s2_samples import flm_2d_to_hp
from s2fft.utils import signal_generator


@pytest.fixture
def flm_generator(numpy_rng) -> Callable[[...], np.ndarray]:
    # Import s2fft (and indirectly numpy) locally to avoid
    # `RuntimeWarning: numpy.ndarray size changed` when importing at module level
    # import s2fft as s2f
    # from s2fft.utils import signal_generator
    def generate_flm(L: int, healpy_ordering: bool = False, **keywords: Any) -> np.ndarray:
        flm = signal_generator.generate_flm(numpy_rng, L, **keywords)
        if healpy_ordering:
            flm = flm_2d_to_hp(flm, L)
        return flm

    return generate_flm
