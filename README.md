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

