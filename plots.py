"""
plots.py

Visualization functions for the nonlinear-toolkit: phase-space
attractors, the H x C complexity plane (with theoretical boundary
curves), bifurcation diagrams, and Lyapunov divergence curves.
"""

import numpy as np
import matplotlib.pyplot as plt
import math

from embedding import embed
from entropy_complexity import entropy_from_distribution, complexity_from_distribution


# ---------------------------------------------------------------
# Attractor plots
# ---------------------------------------------------------------

def plot_attractor(series=None, state=None, dim=3, tau=1,
                    title=None, ax=None):
    """
    Plot a phase-space attractor.

    Either pass `state` directly (an (N, 3) array, e.g. from
    systems.lorenz), or pass a 1D `series` to reconstruct the
    attractor via time-delay embedding (dim=2 or dim=3 supported
    for plotting).

    Parameters
    ----------
    series : array-like, shape (N,), optional
        1D time series to embed (ignored if `state` is given).
    state : array-like, shape (N, 3), optional
        Directly-simulated multivariate state (e.g. Lorenz x,y,z).
    dim : int
        Embedding dimension to use if reconstructing from `series`
        (2 or 3).
    tau : int
        Embedding delay to use if reconstructing from `series`.
    title : str, optional
    ax : matplotlib Axes, optional
        If given, plot into this axes (must be 3D if dim/state is
        3D). Otherwise a new figure is created.

    Returns
    -------
    ax : matplotlib Axes
    """
    if state is not None:
        state = np.asarray(state)
        if ax is None:
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection="3d")
        ax.plot(state[:, 0], state[:, 1], state[:, 2],
                lw=0.5, color="C0")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    elif series is not None:
        vectors = embed(series, dim=dim, tau=tau)
        if dim == 2:
            if ax is None:
                fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(vectors[:, 0], vectors[:, 1], lw=0.5, color="C0")
            ax.set_xlabel("x(t)")
            ax.set_ylabel(f"x(t+{tau})")
        elif dim == 3:
            if ax is None:
                fig = plt.figure(figsize=(6, 5))
                ax = fig.add_subplot(111, projection="3d")
            ax.plot(vectors[:, 0], vectors[:, 1], vectors[:, 2],
                    lw=0.5, color="C0")
            ax.set_xlabel("x(t)")
            ax.set_ylabel(f"x(t+{tau})")
            ax.set_zlabel(f"x(t+{2*tau})")
        else:
            raise ValueError("plot_attractor only supports dim=2 or dim=3.")
    else:
        raise ValueError("Provide either `series` or `state`.")

    if title:
        ax.set_title(title)
    return ax


# ---------------------------------------------------------------
# H x C complexity plane
# ---------------------------------------------------------------

def _cmin_curve(N_states, n_points=300):
    """
    Minimum-complexity boundary curve C_min(H), via the family
    P = (p, (1-p)/(N-1), ..., (1-p)/(N-1)), p in [1/N, 1]
    (Martin, Plastino & Rosso, 2003).
    """
    ps = np.linspace(1.0 / N_states, 1.0, n_points)
    Hs = np.empty(n_points)
    Cs = np.empty(n_points)
    for i, p in enumerate(ps):
        probs = np.full(N_states, (1 - p) / (N_states - 1))
        probs[0] = p
        Hs[i] = entropy_from_distribution(probs, N_states)
        Cs[i] = complexity_from_distribution(probs, N_states)
    order = np.argsort(Hs)
    return Hs[order], Cs[order]


def _cmax_curve(N_states, n_points_per_segment=60):
    """
    Maximum-complexity boundary curve C_max(H), via the family
    (for m = 1..N-1): top m states share probability p_j, the
    remaining N-m states share the rest equally
    (Martin, Plastino & Rosso, 2003).
    """
    Hs, Cs = [], []
    for m in range(1, N_states):
        lo, hi = 1.0 / N_states, 1.0 / m
        for pj in np.linspace(lo, hi, n_points_per_segment):
            probs = np.zeros(N_states)
            probs[:m] = pj
            remaining = 1 - m * pj
            if N_states - m > 0:
                probs[m:] = remaining / (N_states - m)
            if remaining < -1e-9:
                continue
            Hs.append(entropy_from_distribution(probs, N_states))
            Cs.append(complexity_from_distribution(probs, N_states))
    Hs, Cs = np.array(Hs), np.array(Cs)
    order = np.argsort(Hs)
    return Hs[order], Cs[order]


def plot_hc_plane(H_values, C_values, labels=None, dx=4,
                   show_boundaries=True, ax=None, title=None):
    """
    Plot points on the permutation entropy x statistical complexity
    (H x C) plane, optionally with the theoretical min/max
    complexity boundary curves for embedding dimension dx.

    Parameters
    ----------
    H_values, C_values : array-like, same length
        The (H, C) points to plot (e.g. one per system/time series).
    labels : list of str, optional
        Labels for each point (e.g. system names), shown in a legend.
    dx : int
        Embedding dimension used to compute H_values/C_values -
        determines N_states = dx! for the boundary curves.
    show_boundaries : bool
        If True, overlay the theoretical C_min(H)/C_max(H) envelope.
    ax : matplotlib Axes, optional

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    if show_boundaries:
        N_states = math.factorial(dx)
        Hmin, Cmin = _cmin_curve(N_states)
        Hmax, Cmax = _cmax_curve(N_states)
        ax.plot(Hmin, Cmin, "k-", lw=1, label="C_min(H)")
        ax.plot(Hmax, Cmax, "k-", lw=1, label="C_max(H)")

    H_values = np.atleast_1d(H_values)
    C_values = np.atleast_1d(C_values)

    if labels is not None:
        for h, c, lab in zip(H_values, C_values, labels):
            ax.scatter(h, c, s=50, label=lab)
    else:
        ax.scatter(H_values, C_values, s=50, color="C1")

    ax.set_xlabel("Permutation Entropy, H")
    ax.set_ylabel("Statistical Complexity, C")
    ax.set_xlim(-0.02, 1.02)
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    return ax


# ---------------------------------------------------------------
# Bifurcation diagram
# ---------------------------------------------------------------

def plot_bifurcation(map_func, param_name, param_range,
                      n_iterations=200, n_discard=200, x0=0.5,
                      ax=None, title=None):
    """
    Classic bifurcation diagram: for each parameter value, iterate
    the map, discard the transient, and plot the remaining values.

    Parameters
    ----------
    map_func : callable
        A generator function matching systems.py's signature, e.g.
        systems.logistic_map, taking (param_name=..., x0=..., n=...,
        discard=...) and returning an array of length n.
    param_name : str
        Name of the map's control parameter keyword (e.g. "r").
    param_range : array-like
        Values of the control parameter to sweep over.
    n_iterations : int
        Number of post-transient points to plot per parameter value.
    n_discard : int
        Number of initial iterations discarded as transient.
    x0 : float
        Initial condition (same for every parameter value).
    ax : matplotlib Axes, optional

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    param_range = np.asarray(param_range)
    for p in param_range:
        kwargs = {param_name: p, "x0": x0,
                  "n": n_iterations, "discard": n_discard}
        x = map_func(**kwargs)
        ax.plot(np.full(n_iterations, p), x, ",k", alpha=0.5)

    ax.set_xlabel(param_name)
    ax.set_ylabel("x")
    if title:
        ax.set_title(title)
    return ax


# ---------------------------------------------------------------
# Lyapunov divergence curve
# ---------------------------------------------------------------

def plot_divergence_curve(divergence_curve, fit_range=None,
                           lyap_exp=None, ax=None, title=None):
    """
    Plot the averaged log-divergence curve from
    lyapunov.rosenstein_lyapunov, with the fitted linear region
    highlighted - useful for visually checking that max_iter/
    fit_range were chosen before the curve saturates.

    Parameters
    ----------
    divergence_curve : array-like
        The divergence_curve array returned by rosenstein_lyapunov.
    fit_range : tuple(int, int), optional
        The (start, end) range that was fit - drawn as a highlighted
        segment with its fitted line.
    lyap_exp : float, optional
        The estimated Lyapunov exponent (slope), shown in the legend.
    ax : matplotlib Axes, optional

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    k = np.arange(len(divergence_curve))
    ax.plot(k, divergence_curve, "o-", color="C0", ms=3,
            label="<ln d(k)>")

    if fit_range is not None:
        start, end = fit_range
        x_fit = np.arange(start, end)
        y_fit = divergence_curve[start:end]
        mask = ~np.isnan(y_fit)
        x_fit, y_fit = x_fit[mask], y_fit[mask]
        if len(x_fit) >= 2:
            slope, intercept = np.polyfit(x_fit, y_fit, 1)
            label = f"fit (slope={slope:.4f})"
            if lyap_exp is not None:
                label = f"fit (lambda={lyap_exp:.4f})"
            ax.plot(x_fit, slope * x_fit + intercept, "r-",
                    lw=2, label=label)

    ax.set_xlabel("Time step, k")
    ax.set_ylabel("<ln d(k)>")
    ax.legend()
    if title:
        ax.set_title(title)
    return ax


if __name__ == "__main__":
    from systems import logistic_map, lorenz
    from entropy_complexity import h_c
    from lyapunov import rosenstein_lyapunov

    # 1. Attractor plots
    fig1, axes1 = plt.subplots(1, 2, figsize=(11, 5))
    x_chaos = logistic_map(r=4.0, n=2000, discard=200)
    plot_attractor(series=x_chaos, dim=2, title="Logistic map (r=4.0)",
                    ax=axes1[0])
    t, state = lorenz(n_points=5000)
    fig2 = plt.figure(figsize=(6, 5))
    ax_lorenz = fig2.add_subplot(111, projection="3d")
    plot_attractor(state=state, title="Lorenz attractor", ax=ax_lorenz)

    # 2. H x C plane with a few systems plotted together
    x_periodic = logistic_map(r=3.2, n=5000, discard=200)
    x_random = np.random.rand(5000)
    Hs, Cs, labs = [], [], []
    for x, lab in [(x_chaos, "chaotic (r=4.0)"),
                   (x_periodic, "periodic (r=3.2)"),
                   (x_random, "random noise")]:
        H, C = h_c(x)
        Hs.append(H)
        Cs.append(C)
        labs.append(lab)
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    plot_hc_plane(Hs, Cs, labels=labs, dx=4, ax=ax3,
                  title="H x C plane")

    # 3. Bifurcation diagram
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    plot_bifurcation(logistic_map, "r", np.linspace(2.5, 4.0, 800),
                      ax=ax4, title="Logistic map bifurcation diagram")

    # 4. Lyapunov divergence curve
    lyap, curve = rosenstein_lyapunov(x_chaos, dim=2, tau=1, max_iter=10)
    fig5, ax5 = plt.subplots(figsize=(6, 5))
    plot_divergence_curve(curve, fit_range=(0, 10), lyap_exp=lyap,
                           ax=ax5, title="Lyapunov divergence (logistic, r=4.0)")

    plt.show()
    print("All plots generated. Check the figure windows.")