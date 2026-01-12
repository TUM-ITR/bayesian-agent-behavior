# Bayesian Parameter Identification and Behaviour Prediction  
## Differential Games (LQ and Nonlinear Examples)

This repository contains MATLAB code accompanying a research paper on  
**online Bayesian parameter identification and behaviour prediction in differential games**.

We consider two-player, continuous-time differential games and address two main objectives:

1. **Online Bayesian parameter estimation**
   - Gaussian priors on unknown cost and value-function parameters
   - Recursive Bayesian regression using noisy online observations

2. **Behaviour prediction**
   - Use posterior parameter estimates to predict future state and control trajectories
   - Propagate uncertainty forward via Monte Carlo simulation to obtain predictive distributions

The repository includes:
- a **linear–quadratic (LQ)** reference-tracking game, and  
- a **nonlinear 1D** differential game used as a conceptual and numerical illustration in the paper.

The focus of the code is **clarity and reproducibility**, rather than toolbox-style generality.

---

## Repository structure

```text
.
├── example_LQ_bayes_identification_prediction.m
│   Main LQ differential game example
│
├── example_NL_bayes_identification_prediction.m
│   1D nonlinear differential game example (paper example)
│
├── core/
│   Core algorithms (Riccati solvers, value-function computation, etc.)
│
├── utils/
│   Utility functions (sampling, Bayesian updates, regressors, helpers)
│
├── visualization/
│   Plotting and visualization routines
│
└── README.md
```
EXAMPLE 1: LQ DIFFERENTIAL GAME

File:
example_LQ_bayes_identification_prediction.m

Description:

This script simulates a two-player continuous-time linear–quadratic
differential game with reference tracking.

It demonstrates:

Online Bayesian estimation of value-function parameters and diagonal cost
matrices

Reconstruction of feedback controllers from posterior means

Behaviour prediction:

future state trajectories

future control trajectories

uncertainty propagation via Monte Carlo sampling

Key features:

Linear dynamics with two control channels

Gaussian priors on unknown cost parameters

Recursive Bayesian regression

Posterior-based prediction of closed-loop behaviour

Notes:

The closed-loop ODE is written in terms of the deviation from the reference
x_dev(t) = x(t) - x_ref(t)
and this deviation is used implicitly throughout the script.

Cross-control cost terms are set to zero (R12 = R21 = 0).

EXAMPLE 2: NONLINEAR 1D DIFFERENTIAL GAME (PAPER EXAMPLE)

File:
example_NL_bayes_identification_prediction.m

Description:

This script reproduces the 1D nonlinear two-player differential game used in
the paper.

The nonlinear dynamics take the form:
x_dot = f(x) + g1(x) u1 + g2(x) u2

The example mirrors the objectives of the LQ case, while emphasizing conceptual
clarity.

What this example shows:

Offline computation of a high-order reference solution (ground truth)

Online Bayesian estimation of:

truncated value-function weights

scalar state-cost parameters

Posterior analysis:

parameter convergence

relative estimation errors

uncertainty-aware value-function reconstruction (mean ± 2σ)

Notes:

Although the system is scalar, the formulation is compatible with a matrix
viewpoint: g1(x) and g2(x) are treated as 1×1 matrices.

Constant factors in the optimal control laws depend on the chosen feature
normalization and match the paper’s formulation.

REQUIREMENTS

MATLAB (tested with recent versions)

Optimization Toolbox (required for lsqnonlin)

No third-party dependencies

HOW TO RUN

Clone the repository and add it to your MATLAB path.

Run one of the example scripts:

example_LQ_bayes_identification_prediction

or

example_NL_bayes_identification_prediction

Figures will be generated automatically. Plotting can be disabled via flags
inside the scripts.

INTENDED AUDIENCE

This code is intended for:

readers of the accompanying paper,

researchers working on learning and inference in differential games,

users interested in Bayesian identification and prediction in continuous-time
control.

The focus is clarity and reproducibility, not toolbox generality.

CITATION

If you use this code in your work, please cite the associated paper:

[ Citation to be added ]

CONTACT

For questions or issues related to the code, please contact the authors.
