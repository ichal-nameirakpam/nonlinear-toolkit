"""
lyapunov.py

Largest Lyapunov exponent estimation via the Rosenstein et al.
(1993) method - works on any time series (no known governing
equation required), using time-delay phase space reconstruction.

Reference:
Rosenstein, M.T., Collins, J.J., De Luca, C.J. (1993).
"A practical method for calculating largest Lyapunov exponents
from small data sets." Physica D, 65(1-2), 117-134.
"""

import numpy as np


def _embed(series, dim, tau):
    """
    Time-delay embedding: reconstruct phase space vectors from a
    1D time series.

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


def rosenstein_lyapunov(series, dim=4, tau=1, min_tsep=None,
                         max_iter=10, fit_range=None):
    """
    Estimate the largest Lyapunov exponent via Rosenstein's method.

    Steps:
    1. Reconstruct phase space via time-delay embedding.
    2. For each point, find its nearest neighbor (excluding points
       that are temporally close, to avoid trivial autocorrelation).
    3. Track how the (log) distance between each pair grows over
       time steps.
    4. Average the log-distance growth curve over all pairs.
    5. The slope of the initial linear region of this curve is the
       largest Lyapunov exponent.

    Parameters
    ----------
    series : array-like, shape (N,)
    dim : int
        Embedding dimension.
    tau : int
        Embedding delay.
    min_tsep : int or None
        Minimum temporal separation for a valid nearest neighbor
        (avoids picking a trivially close-in-time point). Defaults
        to dim * tau if not given.
    max_iter : int
        Maximum number of future steps to track divergence over.
        Kept small (default 10) because bounded low-dimensional
        systems saturate quickly once trajectories diverge across
        the attractor - averaging past that point corrupts the
        slope estimate.
    fit_range : tuple(int, int) or None
        (start, end) indices into the divergence curve over which
        to fit the slope. If None, uses the first third of the
        available curve (a reasonable default "linear region").

    Returns
    -------
    lyap_exp : float
        Estimated largest Lyapunov exponent (per time step).
        Positive => chaotic, ~0 => periodic/quasi-periodic,
        negative => stable/converging.
    divergence_curve : ndarray
        The averaged log-distance curve actually used in the fit
        (useful for plotting/diagnostics).
    """
    vectors = _embed(series, dim, tau)
    M = len(vectors)

    if min_tsep is None:
        min_tsep = dim * tau


    # Step 1: find nearest neighbor for each point, excluding
    # temporally close points.
    nn_indices = np.empty(M, dtype=int)
    for i in range(M):
        dists = np.linalg.norm(vectors - vectors[i], axis=1)
        dists[max(0, i - min_tsep): i + min_tsep + 1] = np.inf
        nn_indices[i] = np.argmin(dists)

    # Step 2: track divergence of each pair over max_iter steps.
    divergence_sums = np.zeros(max_iter)
    divergence_counts = np.zeros(max_iter)

    for k in range(max_iter):
        valid = (np.arange(M) + k < M) & (nn_indices + k < M)
        idx_i = np.arange(M)[valid]
        idx_j = nn_indices[valid]

        d = np.linalg.norm(
            vectors[idx_i + k] - vectors[idx_j + k], axis=1
        )
        d = np.maximum(d, 1e-12)  # floor to avoid log(0) for
                                    # near-identical trajectories
        divergence_sums[k] = np.sum(np.log(d))
        divergence_counts[k] = len(d)

    valid_k = divergence_counts > 0
    divergence_curve = np.full(max_iter, np.nan)
    divergence_curve[valid_k] = (
        divergence_sums[valid_k] / divergence_counts[valid_k]
    )

    # Step 3: fit a line to the initial (roughly linear) region.
    if fit_range is None:
        fit_range = (0, max_iter)

    start, end = fit_range
    y = divergence_curve[start:end]
    x = np.arange(start, end)
    mask = ~np.isnan(y)
    x, y = x[mask], y[mask]

    if len(x) < 2:
        raise ValueError(
            "Not enough valid points in fit_range to estimate slope. "
            "Try a longer series or a smaller fit_range."
        )

    slope, intercept = np.polyfit(x, y, 1)
    lyap_exp = slope

    return lyap_exp, divergence_curve


if __name__ == "__main__":
    from systems import logistic_map

    # Chaotic logistic map: analytical Lyapunov exponent at r=4.0
    # is exactly ln(2) ~= 0.6931 (known closed-form result)
    x_chaos = logistic_map(r=4.0, n=3000, discard=200)
    lyap, curve = rosenstein_lyapunov(x_chaos, dim=2, tau=1)
    print(f"Logistic map (r=4.0): estimated lambda = {lyap:.4f} "
          f"(analytical = {np.log(2):.4f})")

    # Periodic logistic map (r=3.2): Lyapunov exponent should be
    # negative or near zero (converges to a stable 2-cycle)
    x_periodic = logistic_map(r=3.2, n=3000, discard=200)
    lyap_p, _ = rosenstein_lyapunov(x_periodic, dim=2, tau=1)
    print(f"Logistic map (r=3.2): estimated lambda = {lyap_p:.4f}")