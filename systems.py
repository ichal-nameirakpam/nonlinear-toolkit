"""
systems.py

Generators for classic nonlinear dynamical systems used to test
chaos and complexity measures (entropy, statistical complexity,
Lyapunov exponents, etc.).

Discrete maps return a 1D numpy array of length n.
Continuous systems (Lorenz) return a time array and a state array,
obtained via numerical integration.
"""

import numpy as np
from scipy.integrate import solve_ivp


def logistic_map(r=4.0, x0=0.4, n=1000, discard=0):
    """
    Logistic map: x_{t+1} = r * x_t * (1 - x_t)

    Parameters
    ----------
    r : float
        Growth rate parameter. r=4.0 gives fully chaotic behavior.
    x0 : float
        Initial condition, in (0, 1).
    n : int
        Number of points to return.
    discard : int
        Number of initial iterations to discard (transient removal).

    Returns
    -------
    x : ndarray, shape (n,)
    """
    total = n + discard
    x = np.empty(total)
    x[0] = x0
    for t in range(total - 1):
        x[t + 1] = r * x[t] * (1 - x[t])
    return x[discard:]


def sine_map(mu=1.0, x0=0.4, n=1000, discard=0):
    """
    Sine map: x_{t+1} = mu * sin(pi * x_t)

    Parameters
    ----------
    mu : float
        Control parameter, in [0, 1]. mu close to 1 gives chaos.
    x0 : float
        Initial condition, in (0, 1).
    n : int
        Number of points to return.
    discard : int
        Number of initial iterations to discard.

    Returns
    -------
    x : ndarray, shape (n,)
    """
    total = n + discard
    x = np.empty(total)
    x[0] = x0
    for t in range(total - 1):
        x[t + 1] = mu * np.sin(np.pi * x[t])
    return x[discard:]


def henon_map(a=1.4, b=0.3, x0=0.0, y0=0.0, n=1000, discard=0):
    """
    Henon map (2D discrete chaotic map):
        x_{t+1} = 1 - a * x_t^2 + y_t
        y_{t+1} = b * x_t

    Parameters
    ----------
    a, b : float
        Standard chaotic parameters: a=1.4, b=0.3.
    x0, y0 : float
        Initial conditions.
    n : int
        Number of points to return.
    discard : int
        Number of initial iterations to discard.

    Returns
    -------
    x, y : ndarray, shape (n,) each
    """
    total = n + discard
    x = np.empty(total)
    y = np.empty(total)
    x[0], y[0] = x0, y0
    for t in range(total - 1):
        x[t + 1] = 1 - a * x[t] ** 2 + y[t]
        y[t + 1] = b * x[t]
    return x[discard:], y[discard:]


def lorenz(sigma=10.0, rho=28.0, beta=8.0 / 3.0,
           initial_state=(1.0, 1.0, 1.0), t_span=(0, 50), n_points=5000):
    """
    Lorenz system (continuous, integrated with scipy):
        dx/dt = sigma * (y - x)
        dy/dt = x * (rho - z) - y
        dz/dt = x * y - beta * z

    Default parameters (sigma=10, rho=28, beta=8/3) are the classic
    chaotic regime.

    Parameters
    ----------
    sigma, rho, beta : float
        Lorenz system parameters.
    initial_state : tuple of 3 floats
        Initial (x, y, z).
    t_span : tuple of 2 floats
        (t_start, t_end) for integration.
    n_points : int
        Number of evenly spaced time points to sample.

    Returns
    -------
    t : ndarray, shape (n_points,)
    state : ndarray, shape (n_points, 3)
        Columns are x, y, z.
    """
    def deriv(t, state):
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return [dx, dy, dz]

    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(deriv, t_span, initial_state, t_eval=t_eval,
                     method="RK45", rtol=1e-9, atol=1e-9)
    return sol.t, sol.y.T


if __name__ == "__main__":
    # Quick sanity check when run directly
    x = logistic_map(n=10, discard=0)
    print("Logistic map first 10 values:", x)

    xh, yh = henon_map(n=10)
    print("Henon map x first 10 values:", xh)

    t, state = lorenz(n_points=10)
    print("Lorenz t shape:", t.shape, "state shape:", state.shape)

