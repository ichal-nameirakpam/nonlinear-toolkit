"""
entropy_complexity.py

Permutation entropy (H) and statistical complexity (C) via the
Bandt-Pompe ordinal pattern method (Bandt & Pompe, 2002), combined
with the Jensen-Shannon statistical complexity of Lopez-Ruiz,
Mancini & Calbet (1995) / Rosso et al. (2007).

This reimplements the core of what ordpy provides, from scratch,
for a time series x(t).
"""

import numpy as np
import math
from itertools import permutations


def ordinal_patterns(series, dx=4, taux=1):
    """
    Convert a time series into its sequence of ordinal (Bandt-Pompe)
    patterns.

    Each window of dx consecutive points (spaced by taux) is mapped
    to the permutation describing the relative order of its values.

    Parameters
    ----------
    series : array-like, shape (N,)
    dx : int
        Embedding dimension (pattern length). dx! possible patterns.
    taux : int
        Embedding delay (time lag between points in a pattern).

    Returns
    -------
    patterns : ndarray of shape (n_windows, dx)
        Each row is a permutation of range(dx), e.g. [2, 0, 1, 3].
    """
    series = np.asarray(series)
    N = len(series)
    n_windows = N - (dx - 1) * taux
    if n_windows <= 0:
        raise ValueError(
            f"Series too short for dx={dx}, taux={taux}: "
            f"need at least {(dx - 1) * taux + 1} points, got {N}."
        )

    patterns = np.empty((n_windows, dx), dtype=int)
    for i in range(n_windows):
        window = series[i: i + dx * taux: taux]
        patterns[i] = np.argsort(window)
    return patterns


def _pattern_distribution(series, dx=4, taux=1):
    """
    Internal helper: return the empirical probability distribution
    over the dx! possible ordinal patterns, as an array of length dx!
    (probabilities for patterns that never occur are 0).
    """
    patterns = ordinal_patterns(series, dx=dx, taux=taux)

    all_perms = list(permutations(range(dx)))
    perm_to_index = {perm: i for i, perm in enumerate(all_perms)}

    counts = np.zeros(len(all_perms))
    for row in patterns:
        counts[perm_to_index[tuple(row)]] += 1

    probs = counts / counts.sum()
    return probs


def permutation_entropy(series, dx=4, taux=1, normalize=True):
    """
    Permutation entropy H of a time series (Bandt & Pompe, 2002).

    H = -sum(p_i * log(p_i)) over the observed ordinal patterns,
    optionally normalized by log(dx!) so that H is in [0, 1].

    Parameters
    ----------
    series : array-like, shape (N,)
    dx : int
        Embedding dimension.
    taux : int
        Embedding delay.
    normalize : bool
        If True (default), divide by log(dx!) so H in [0, 1].
        H=0: fully predictable/periodic. H=1: fully random.

    Returns
    -------
    H : float
    """
    probs = _pattern_distribution(series, dx=dx, taux=taux)
    probs_nonzero = probs[probs > 0]

    H = -np.sum(probs_nonzero * np.log(probs_nonzero))

    if normalize:
        H_max = np.log(math.factorial(dx))
        H = H / H_max
    return H


def statistical_complexity(series, dx=4, taux=1):
    """
    Jensen-Shannon statistical complexity C of a time series
    (Lopez-Ruiz, Mancini & Calbet 1995; Rosso et al. 2007 MPR form).

    C = Q_J * H_normalized

    where Q_J is the Jensen-Shannon "disequilibrium" between the
    pattern distribution P and the uniform distribution P_e,
    normalized by its maximum possible value Q_max.

    Parameters
    ----------
    series : array-like, shape (N,)
    dx : int
        Embedding dimension.
    taux : int
        Embedding delay.

    Returns
    -------
    C : float
        Statistical complexity, roughly in [0, C_max] with
        C_max < 1. C=0 for both fully ordered and fully random
        series; C peaks for intermediate, structured-but-not-
        periodic dynamics (e.g. chaos).
    """
    N_states = math.factorial(dx)
    probs = _pattern_distribution(series, dx=dx, taux=taux)
    H_norm = permutation_entropy(series, dx=dx, taux=taux, normalize=True)

    p_uniform = np.full(N_states, 1.0 / N_states)

    # Jensen-Shannon divergence between P and uniform P_e
    p_mix = 0.5 * (probs + p_uniform)

    def shannon(p):
        p_nz = p[p > 0]
        return -np.sum(p_nz * np.log(p_nz))

    S_mix = shannon(p_mix)
    S_p = shannon(probs)
    S_uniform = shannon(p_uniform)  # = log(N_states)

    J = S_mix - 0.5 * S_p - 0.5 * S_uniform

    # Normalization constant Q_max (Rosso et al. 2007)
    Q_max = -0.5 * (
        ((N_states + 1) / N_states) * np.log(N_states + 1)
        + np.log(N_states)
        - 2 * np.log(2 * N_states)
    )

    Q_J = J / Q_max
    C = Q_J * H_norm
    return C


def h_c(series, dx=4, taux=1):
    """
    Convenience function: return (H, C) together for a series.
    """
    H = permutation_entropy(series, dx=dx, taux=taux, normalize=True)
    C = statistical_complexity(series, dx=dx, taux=taux)
    return H, C


if __name__ == "__main__":
    from systems import logistic_map, henon_map

    # Fully chaotic logistic map (r=4.0) should give high H, moderate C
    x_chaos = logistic_map(r=4.0, n=5000, discard=100)
    H, C = h_c(x_chaos)
    print(f"Logistic map (r=4.0, chaotic): H = {H:.4f}, C = {C:.4f}")

    # Periodic logistic map (r=3.2, period-2 cycle) should give low H, low C
    x_periodic = logistic_map(r=3.2, n=5000, discard=100)
    H, C = h_c(x_periodic)
    print(f"Logistic map (r=3.2, periodic): H = {H:.4f}, C = {C:.4f}")

    # Random noise should give H close to 1, C close to 0
    x_random = np.random.rand(5000)
    H, C = h_c(x_random)
    print(f"Random noise: H = {H:.4f}, C = {C:.4f}")