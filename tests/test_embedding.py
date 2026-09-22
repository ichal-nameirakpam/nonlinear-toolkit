import numpy as np
from systems import lorenz
from embedding import embed, optimal_tau, optimal_dim


def test_embed_shape():
    series = np.arange(100)
    vectors = embed(series, dim=3, tau=2)
    assert vectors.shape == (100 - 2 * 2, 3)


def test_embed_too_short_raises():
    series = np.arange(5)
    try:
        embed(series, dim=4, tau=3)
        assert False, "Expected ValueError for too-short series"
    except ValueError:
        pass


def test_lorenz_optimal_dim_is_three():
    t, state = lorenz(n_points=5000, t_span=(0, 50))
    x = state[:, 0]
    tau, _ = optimal_tau(x, max_tau=30)
    dim, _ = optimal_dim(x, tau=tau, max_dim=8)
    # Lorenz's true phase-space dimension is 3
    assert dim == 3
