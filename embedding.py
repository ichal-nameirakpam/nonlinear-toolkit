"""
embedding.py

Time-delay phase space reconstruction (Takens' embedding theorem),
plus two standard methods for choosing good embedding parameters
from data alone:

  - mutual_information / optimal_tau: choose the embedding delay
    tau via the first local minimum of the time-delayed mutual
    information (Fraser & Swinney, 1986).

  - false_nearest_neighbors / optimal_dim: choose the embedding
    dimension dim via the false nearest neighbors method
    (Kennel, Brown & Abarbanel, 1992).
"""

import numpy as np


def embed(series, dim, tau):
    """
    Time-delay embedding: reconstruct phase space vectors from a
    1D time series using Takens' method.

    Parameters
    ----------
    series : array-like, shape (N,)
    dim : int
        Embedding dimension.
    tau : int
        Time delay.

    Returns
    -------
    vectors : ndarray, shape (M, dim)
        M = N - (dim - 1) * tau reconstructed phase-space vectors.
    """
    series = np.asarray(series)
    N = len(series)
    M = N - (dim - 1) * tau
    if M <= 0:
        raise ValueError(
            f"Series too short for dim={dim}, tau={tau}: "
            f"need at least {(dim - 1) * tau + 1} points, got {N}."
        )
    vectors = np.empty((M, dim))
    for i in range(dim):
        vectors[:, i] = series[i * tau: i * tau + M]
    return vectors


def mutual_information(series, max_tau=30, bins=16):
    """
    Time-delayed mutual information I(tau) between series[t] and
    series[t+tau], for tau = 1..max_tau, estimated via histogram
    binning.

    Returns
    -------
    mi_values : ndarray, shape (max_tau,)
        mi_values[k] is I(tau = k+1).
    """
    series = np.asarray(series)
    mi_values = np.empty(max_tau)
    hist_range = [[series.min(), series.max()], [series.min(), series.max()]]

    for tau in range(1, max_tau + 1):
        x = series[:-tau]
        y = series[tau:]
        c_xy, _, _ = np.histogram2d(x, y, bins=bins, range=hist_range)
        p_xy = c_xy / c_xy.sum()
        p_x = p_xy.sum(axis=1)
        p_y = p_xy.sum(axis=0)
        px_py = np.outer(p_x, p_y)
        nonzero = p_xy > 0
        mi = np.sum(p_xy[nonzero] * np.log(p_xy[nonzero] / px_py[nonzero]))
        mi_values[tau - 1] = mi

    return mi_values


def optimal_tau(series, max_tau=30, bins=16):
    """
    Choose the embedding delay as the first local minimum of the
    mutual information curve (Fraser & Swinney, 1986). Falls back
    to the global minimum if no local minimum is found.

    Returns
    -------
    tau : int
    mi_values : ndarray
        The full mutual information curve (for plotting/diagnostics).
    """
    mi = mutual_information(series, max_tau=max_tau, bins=bins)
    for i in range(1, len(mi) - 1):
        if mi[i] < mi[i - 1] and mi[i] < mi[i + 1]:
            return i + 1, mi
    return int(np.argmin(mi)) + 1, mi


def false_nearest_neighbors(series, tau, max_dim=10, rtol=15.0, atol=2.0):
    """
    False nearest neighbors fraction for embedding dimensions
    1..max_dim (Kennel, Brown & Abarbanel, 1992).

    For each point, compares its nearest neighbor in dimension d
    against dimension d+1: if the neighbor distance grows too much
    when adding a dimension, it was a "false" neighbor - an
    artifact of an embedding that's too low-dimensional to unfold
    the true dynamics.

    Parameters
    ----------
    tau : int
        Embedding delay to use (choose via optimal_tau first).
    rtol : float
        Relative distance threshold (standard value: 10-15).
    atol : float
        Absolute distance threshold, in units of the series std
        (standard value: ~2).

    Returns
    -------
    fnn_fractions : ndarray, shape (max_dim,)
    """
    series = np.asarray(series)
    std = np.std(series)
    fnn_fractions = np.empty(max_dim)

    for dim in range(1, max_dim + 1):
        vec_d = embed(series, dim, tau)
        vec_d1 = embed(series, dim + 1, tau)
        M = len(vec_d1)
        vec_d = vec_d[:M]

        n_false = 0
        for i in range(M):
            dists = np.linalg.norm(vec_d - vec_d[i], axis=1)
            dists[i] = np.inf
            j = np.argmin(dists)
            Rd = dists[j]
            if Rd == 0:
                continue
            extra_dist = abs(vec_d1[i, -1] - vec_d1[j, -1])
            Rd1 = np.sqrt(Rd ** 2 + extra_dist ** 2)
            if (Rd1 / Rd) > rtol or (Rd1 / std) > atol:
                n_false += 1

        fnn_fractions[dim - 1] = n_false / M

    return fnn_fractions


def optimal_dim(series, tau, max_dim=10, threshold=0.01, **kwargs):
    """
    Choose the embedding dimension as the first dim where the false
    nearest neighbors fraction drops below `threshold`.

    Returns
    -------
    dim : int
    fnn_fractions : ndarray
        The full FNN curve (for plotting/diagnostics).
    """
    fnn = false_nearest_neighbors(series, tau, max_dim=max_dim, **kwargs)
    for d in range(len(fnn)):
        if fnn[d] < threshold:
            return d + 1, fnn
    return max_dim, fnn


if __name__ == "__main__":
    from systems import lorenz

    t, state = lorenz(n_points=5000, t_span=(0, 50))
    x = state[:, 0]  # use just the x-coordinate, as if it were
                      # all we had measured from a real system

    tau, mi_curve = optimal_tau(x, max_tau=30)
    print(f"Lorenz x(t): optimal tau (first MI minimum) = {tau}")

    dim, fnn_curve = optimal_dim(x, tau=tau, max_dim=8)
    print(f"Lorenz x(t): optimal embedding dimension = {dim} "
          f"(expected ~3, the true Lorenz phase space dimension)")