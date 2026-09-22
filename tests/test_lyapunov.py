import numpy as np
from systems import logistic_map
from lyapunov import rosenstein_lyapunov


def test_chaotic_logistic_matches_analytical():
    x = logistic_map(r=4.0, n=3000, discard=200)
    lyap, _ = rosenstein_lyapunov(x, dim=2, tau=1, max_iter=10)
    # Analytical value is exactly ln(2) ~= 0.6931
    assert abs(lyap - np.log(2)) < 0.05


def test_periodic_logistic_near_zero():
    x = logistic_map(r=3.2, n=3000, discard=200)
    lyap, _ = rosenstein_lyapunov(x, dim=2, tau=1, max_iter=10)
    assert abs(lyap) < 0.05