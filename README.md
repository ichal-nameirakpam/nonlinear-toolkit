# nonlinear-toolkit

A Python toolkit for generating classic chaotic dynamical systems and
characterizing them with information-theoretic and dynamical measures:
permutation entropy, Jensen-Shannon statistical complexity, the largest
Lyapunov exponent, and time-delay phase space reconstruction.

Every method implemented here is validated against a known analytical
or literature result (see **Validation** below) rather than assumed
correct.

## Motivation

Distinguishing chaotic dynamics from noise or periodicity is a
recurring problem across physics, from dynamical systems theory to
signal analysis of real-world data such as gravitational wave strain.
This toolkit implements, from first principles, the core measures used
in that kind of analysis:

- **Permutation entropy (H)** and **statistical complexity (C)**,
  following the Bandt-Pompe ordinal pattern method (Bandt & Pompe,
  2002) and the Jensen-Shannon complexity of Lopez-Ruiz, Mancini &
  Calbet (1995), in the form developed by Rosso et al. (2007). Chaotic
  signals occupy a distinct region of the H x C plane: high entropy
  but nonzero complexity, separating them from both periodic (low H,
  low C) and stochastic (H close to 1, C close to 0) signals.
- **The largest Lyapunov exponent**, estimated directly from a time
  series using the Rosenstein et al. (1993) method, which requires no
  knowledge of the system's governing equations. A positive exponent
  is the classic signature of sensitive dependence on initial
  conditions, the defining property of chaos.
- **Time-delay embedding** (Takens' theorem), with data-driven
  selection of the embedding delay (via mutual information, Fraser &
  Swinney, 1986) and embedding dimension (via false nearest
  neighbors, Kennel, Brown & Abarbanel, 1992).

## Modules

- `systems.py`: generators for the logistic map, sine map, Henon map,
  and the Lorenz system (via numerical integration).
- `entropy_complexity.py`: permutation entropy and Jensen-Shannon
  statistical complexity, computed either from a time series or from
  an arbitrary probability distribution.
- `lyapunov.py`: the largest Lyapunov exponent via Rosenstein's
  method.
- `embedding.py`: time-delay embedding, plus automatic delay and
  dimension selection.
- `plots.py`: phase-space attractor plots, the H x C complexity plane
  with theoretical boundary curves, bifurcation diagrams, and
  Lyapunov divergence curve diagnostics.
- `tests/`: unit tests for every module (15 tests, `pytest`).

## Installation

```bash
git clone https://github.com/ichal-nameirakpam/nonlinear-toolkit.git
cd nonlinear-toolkit
pip install -r requirements.txt
```

## Usage

```python
from systems import logistic_map
from entropy_complexity import h_c
from lyapunov import rosenstein_lyapunov
from plots import plot_hc_plane

x = logistic_map(r=4.0, n=5000, discard=200)

H, C = h_c(x)
print(f"H = {H:.4f}, C = {C:.4f}")

lyap, curve = rosenstein_lyapunov(x, dim=2, tau=1, max_iter=10)
print(f"Largest Lyapunov exponent = {lyap:.4f}")

plot_hc_plane([H], [C], labels=["logistic map (r=4.0)"])
```

## Validation

Every measure implemented here was checked against a known result
before being committed:

| Measure | Test case | Result | Reference value |
|---|---|---|---|
| Largest Lyapunov exponent | Logistic map, r=4.0 | 0.6932 | ln(2) = 0.6931 (analytical) |
| Embedding dimension (FNN) | Lorenz system | 3 | 3 (true phase-space dimension) |
| H, C plane | Chaotic / periodic / random signals | Land correctly inside the theoretical envelope and in the expected regions of the plane | Rosso et al. (2007) |
| Permutation entropy, random noise | White noise | H approx 1.0 | 1.0 (maximal entropy) |

## Background

This toolkit generalizes the ordinal-pattern methodology used in my
MSc dissertation, *Multiscale Information-Theoretic and Dynamical
Characterization of Gravitational Wave Strain Signals*, where
permutation entropy and statistical complexity were applied to
gravitational wave strain data (GW230529) using the `ordpy` library.
Here, the same core methods are reimplemented from scratch and applied
to canonical chaotic systems with known ground truth, as a way of
validating the underlying methodology independently of any external
library.

## References

- Bandt, C. and Pompe, B. (2002). Permutation entropy: a natural
  complexity measure for time series. *Physical Review Letters*, 88(17).
- Lopez-Ruiz, R., Mancini, H.L. and Calbet, X. (1995). A statistical
  measure of complexity. *Physics Letters A*, 209(5-6).
- Rosso, O.A. et al. (2007). Distinguishing noise from chaos.
  *Physical Review Letters*, 99(15).
- Martin, M.T., Plastino, A. and Rosso, O.A. (2003). Statistical
  complexity and disequilibrium. *Physics Letters A*, 311(2-3).
- Rosenstein, M.T., Collins, J.J. and De Luca, C.J. (1993). A
  practical method for calculating largest Lyapunov exponents from
  small data sets. *Physica D*, 65(1-2).
- Fraser, A.M. and Swinney, H.L. (1986). Independent coordinates for
  strange attractors from mutual information. *Physical Review A*,
  33(2).
- Kennel, M.B., Brown, R. and Abarbanel, H.D.I. (1992). Determining
  embedding dimension for phase-space reconstruction using a
  geometrical construction. *Physical Review A*, 45(6).