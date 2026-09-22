import numpy as np
from systems import logistic_map
from entropy_complexity import (
    permutation_entropy, statistical_complexity, h_c
)


def test_random_noise_high_entropy_low_complexity():
    np.random.seed(0)
    x = np.random.rand(5000)
    H, C = h_c(x)
    assert H > 0.95        # near-maximal entropy
    assert C < 0.05         # near-zero complexity


def test_periodic_signal_low_entropy():
    # Logistic map at r=3.2 settles into a stable period-2 cycle,
    # which should give low permutation entropy
    x = logistic_map(r=3.2, n=2000, discard=200)
    H = permutation_entropy(x, dx=4)
    assert H < 0.3


def test_chaotic_logistic_map_moderate_complexity():
    x = logistic_map(r=4.0, n=5000, discard=100)
    H, C = h_c(x)
    # Known regime for chaotic dynamics: high-ish H, nonzero C
    assert 0.5 < H < 1.0
    assert C > 0.1


def test_entropy_normalized_range():
    x = logistic_map(r=4.0, n=2000, discard=100)
    H = permutation_entropy(x, dx=4, normalize=True)
    assert 0.0 <= H <= 1.0