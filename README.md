[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

<p align="center">
  <img src="logo.png" width="360">
</p>

# Bayesian Parameter Identification and Behaviour Prediction in Differential Games 

This repository contains MATLAB code accompanying the paper  
**Online Bayesian Learning of Agent Behavior in Differential Games**  
by *Francesco Bianchin, Robert Lefringhausen, Sandra Hirche*.

The code implements an online Bayesian, game-theoretic approach to **infer and
predict agent behaviour** in multi-agent dynamical systems.  
The key idea is to cast Hamilton–Jacobi–Bellman (HJB) optimality conditions as
**linear-in-parameters residuals**, enabling fast sequential **conjugate Gaussian
updates** and uncertainty-aware prediction from limited, noisy observations
(without history stacks).

---

## Scope and objectives

We consider two-player, continuous-time differential games and address two
main objectives:

1. **Online Bayesian parameter estimation**
   - Gaussian priors on unknown objective parameters (value-function and cost terms)
   - Recursive Bayesian regression using a stream of noisy observations
   - Applicable to nonlinear dynamics and nonquadratic value functions via
     differentiable basis expansions

2. **Behaviour prediction**
   - Use posterior parameter estimates to predict future state and control trajectories
   - Propagate uncertainty forward via Monte Carlo simulation to obtain predictive
     distributions

The repository includes:
- a **linear–quadratic (LQ)** reference-tracking differential game, and  
- a **nonlinear 1D** differential game used as a conceptual and numerical illustration
  in the paper.

The focus is **clarity and reproducibility**, rather than toolbox-style generality.

**Citation**  
Bianchin, F., Lefringhausen, R. and Hirche, S., 2026. *Online Bayesian Learning of
Agent Behavior in Differential Games*. arXiv preprint arXiv:2601.05087.

---

## Mathematical summary (what the code implements)

We consider a two-player continuous-time differential game with dynamics

$$
\dot{x} = f(x) + g_1(x)u_1 + g_2(x)u_2 ,
$$

and infinite-horizon cost functionals

$$
J_i = \int_0^\infty \left( Q_i(x(t)) + u_i(t)^\top R_i\,u_i(t) \right)\,dt,
\qquad i = 1,2 .
$$

At a feedback Nash equilibrium, the value functions satisfy coupled
Hamilton–Jacobi–Bellman optimality conditions together with the stationary
feedback laws

$$
u_i^\star(x)
= -\tfrac{1}{2}\,R_i^{-1} g_i(x)^\top \nabla V_i(x) .
$$

To obtain a tractable inverse problem, the unknown objectives are approximated
using differentiable feature maps,

$$
V_i(x) \approx W_{V_i}^\top \phi_{V_i}(x), \qquad
Q_i(x) \approx W_{Q_i}^\top \phi_{Q_i}(x),
$$

while the quadratic control cost admits an exact linear representation

$$
u_i^\top R_i u_i = W_{R_i}^\top \phi_{R_i}(u_i).
$$

Substituting these representations into the HJB and feedback conditions yields,
at each time step, a **linear regression model**

$$
y_i^{(k)} = \Phi_i^{(k)}\,W_i^- + \eta_i^{(k)}, \qquad
\eta_i^{(k)} \sim \mathcal{N}(0,\Sigma_i),
$$

where $W_i^-$ collects the unknown objective parameters up to scale.
A Gaussian prior $W_i^- \sim \mathcal{N}(m_{0,i}, S_{0,i})$ enables fast
**online conjugate Bayesian updates**.

Posterior uncertainty is propagated forward through the induced feedback
policies using Monte Carlo simulation to obtain predictive distributions over
future states and control inputs.

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

## Example 1: LQ differential game

File:  
example_LQ_bayes_identification_prediction.m

### Description

This script simulates a two-player continuous-time linear–quadratic
differential game with reference tracking.

It demonstrates:
- Online Bayesian estimation of value-function parameters and diagonal cost
  matrices
- Reconstruction of feedback controllers from posterior means
- Behaviour prediction:
  - future state trajectories
  - future control trajectories
  - uncertainty propagation via Monte Carlo sampling

### Notes

- The closed-loop ODE is written in terms of the deviation from the reference:

$$
x_{dev}(t) = x(t) - x_{ref}(t)
$$

  and this deviation is used implicitly throughout the script.

- Cross-control cost terms are set to zero (R12 = R21 = 0).

---

## Example 2: Nonlinear 1D differential game

File:  
example_NL_bayes_identification_prediction.m

### Description

This script reproduces the 1D nonlinear two-player differential game used in
the paper.

The nonlinear dynamics take the form:

$$
\dot{x} = f(x) + g_1(x) u_1 + g_2(x) u_2
$$

---

## Requirements

- MATLAB (tested with recent versions)
- Optimization Toolbox (required for lsqnonlin)
- No third-party dependencies

---

## How to run

1. Clone the repository and add it to your MATLAB path.
2. Run one of the example scripts:

       example_LQ_bayes_identification_prediction

   or

       example_NL_bayes_identification_prediction

3. Figures will be generated automatically. Plotting can be disabled via flags
   inside the scripts.

---

## Intended audience

This code is intended for:
- readers of the accompanying paper,
- researchers working on learning and inference in differential games,
- users interested in Bayesian identification and prediction in continuous-time
  control.

The focus is clarity and reproducibility, not toolbox generality.

---

## Citation

If you use this code in your work, please cite the associated paper:

Bianchin, F., Lefringhausen, R. and Hirche, S., 2026. Online Bayesian Learning of Agent Behavior in Differential Games. arXiv preprint arXiv:2601.05087.

---

## Contact

For questions or issues related to the code, please contact the authors: francesco.bianchin@tum.de
