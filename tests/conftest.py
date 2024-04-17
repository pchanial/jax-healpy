from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope='session')
def numpy_rng() -> np.random.RandomState:
    seed = 0
    return np.random.RandomState(seed)


@pytest.fixture(scope='session')
def data_path() -> Path:
    return Path(__file__).parent / 'data'
