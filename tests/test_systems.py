import numpy as np
from systems import logistic_map, sine_map, henon_map, lorenz


def test_logistic_map_bounded():
    x = logistic_map(r=4.0, n=1000, discard=100)
    assert np.all(x >= 0) and np.all(x <= 1)


def test_logistic_map_fixed_point_r_below_3():
    # For r < 3, logistic map converges to a stable fixed point
    x = logistic_map(r=2.5, n=500, discard=400)
    assert np.allclose(x, x[0], atol=1e-3)


def test_sine_map_bounded():
    x = sine_map(mu=1.0, n=1000, discard=100)
    assert np.all(x >= -1) and np.all(x <= 1)


def test_henon_map_shape():
    x, y = henon_map(n=500)
    assert len(x) == 500
    assert len(y) == 500


def test_lorenz_shape():
    t, state = lorenz(n_points=200)
    assert t.shape == (200,)
    assert state.shape == (200, 3)


def test_lorenz_stays_bounded_short_term():
    # Sanity check: Lorenz trajectory shouldn't blow up to infinity
    # over a short integration window
    t, state = lorenz(t_span=(0, 5), n_points=100)
    assert np.all(np.isfinite(state))
