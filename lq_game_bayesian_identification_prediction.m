%% ============================================================
% BAYESIAN PARAMETER IDENTIFICATION & BEHAVIOUR PREDICTION
% in LQ DIFFERENTIAL GAMES
% ============================================================
% This script simulates a two-player continuous-time linear-quadratic (LQ)
% differential game with reference tracking and:
%
%   (1) Online Bayesian parameter estimation
%       - Gaussian priors on (diagonal) cost parameters
%       - Recursive Bayesian regression on a stream of noisy observations
%
%   (2) Behaviour prediction (future state + input)
%       - Use a posterior snapshot of the learned parameters
%       - Propagate uncertainty forward (Monte Carlo) to obtain predictive
%         distributions over future trajectories and controls
%
% Notes for readers:
%   - The closed-loop ODE is written in terms of the deviation from the reference:
%         x_dev(t) = x(t) - x_ref(t)
%     and the script uses x_dev implicitly throughout (kept in variables like x_true).
%   - Cross terms in control costs are set to zero (R12 = R21 = 0).
% ============================================================

clear all; close all; clc;
rng(4); % reproducibility (global seed)
set(0, 'DefaultFigureRenderer', 'painters');

addpath('core')
addpath('game')
addpath('utils')
addpath('visualization')

%% ============================================================
%  1. SYSTEM DYNAMICS
% ============================================================

A = [ 0  1 -1  0;
      1  0  2  1;
      0 -2  0  1;
      0  1  0 -1];

B1 = 0.5 * [0 1;
            0 0;
            0 0;
            1 0];

B2 = 0.5 * [0 0;
            0 1;
            1 0;
            0 0];

n_x  = size(A, 1);
n_u1 = size(B1, 2);
n_u2 = size(B2, 2);

%% ============================================================
%  2. COST MATRIX PRIORS
% ============================================================

% Mean values (diagonal LQ weights)
Q1_mean  = diag([1, 0.4, 3, 1]);
Q2_mean  = diag([1, 2/3, 1, 2]);
R11_mean = diag([1, 1]);
R22_mean = diag([1, 0.5]);

% Coefficient of variation (relative std)
cv_Q  = 1;
cv_Rd = 1;
eps_floor = 1e-3; % truncation floor for sampling

% Fixed masks (true = fixed/known, false = variable/unknown)
% Assumption used later in the parameter vector:
%   - R11(1,1) and R22(1,1) are treated as known (not estimated)
%   - only the remaining diagonal entry is estimated (for 2x2 R matrices)
mask_R11 = [true, false];
mask_R22 = [true, false];
mask_Q   = [false, false, false, false];

%% ============================================================
%  3. SAMPLE TRUE COST MATRICES
% ============================================================

[Q1,  Q1_mu,  Q1_sig ] = sample_diag_gaussian(Q1_mean,  cv_Q,  eps_floor, mask_Q);
[Q2,  Q2_mu,  Q2_sig ] = sample_diag_gaussian(Q2_mean,  cv_Q,  eps_floor, mask_Q);
[R11, R11_mu, R11_sig] = sample_diag_gaussian(R11_mean, cv_Rd, eps_floor, mask_R11);
[R22, R22_mu, R22_sig] = sample_diag_gaussian(R22_mean, cv_Rd, eps_floor, mask_R22);

% No cross terms
R12 = zeros(size(R11));
R21 = zeros(size(R22));

%% ============================================================
%  4. SOLVE LQ GAME (TRUE AND NOMINAL)
% ============================================================

% True solution
[K2_true, K1_true, P2_true, P1_true] = ...
    basicLQ_game_gajicli(Q2, Q1, R22, R11, R21, R12, A, B2, B1);

% Nominal solution (based on prior means)
[K2_nom, K1_nom, P2_nom, P1_nom] = ...
    basicLQ_game_gajicli(Q2_mean, Q1_mean, R22_mean, R11_mean, R21, R12, A, B2, B1);

% vech operator: stack upper triangular entries of a symmetric matrix
vech = @(M) M(find(triu(true(size(M)))'));

V_true_p1 = vech(P1_true);
V_nom_p1  = vech(P1_nom);
V_true_p2 = vech(P2_true);
V_nom_p2  = vech(P2_nom);

%% ============================================================
%  5. MONTE CARLO EXPECTATION (OPTIONAL)
% ============================================================
% Used to approximate the prior mean/covariance of vech(P_i).

do_MC = true;
if do_MC
    Nmc = 200;
    [V_mc_mean_p1, V_mc_mean_p2, V_mc_cov_p1, V_mc_cov_p2, valid_count] = ...
        montecarlo_P(Q1_mu, Q1_sig, Q2_mu, Q2_sig, R11_mu, R11_sig, R22_mu, R22_sig, ...
                     eps_floor, A, B1, B2, vech, Nmc);
else
    % Fallbacks if MC is disabled (kept minimal to avoid algorithmic changes)
    V_mc_mean_p1 = V_nom_p1;
    V_mc_mean_p2 = V_nom_p2;
    V_mc_cov_p1  = 1e-6 * eye(length(V_nom_p1));
    V_mc_cov_p2  = 1e-6 * eye(length(V_nom_p2));
    valid_count  = 0;
    Nmc = 0;
end

%% ============================================================
%  6. SIMULATION SETUP
% ============================================================

T = 18;
deltaT = 0.01;
x0 = [3; -4; 2; 1.5];

% Reference trajectory (piecewise-constant)
t_switch = [0, 6, 12, 18];
x_ref_list = [
    0   0   0   0;
    1  -2   2   1;
   -2   1   3  -2
]';

% Returns x_ref(t) as a column vector (n_x x 1)
x_ref_fun = @(t) get_reference_vector(t, t_switch, x_ref_list);

t_sim = (0:deltaT:T)';
x_ref_t = cell2mat(arrayfun(@(ti) x_ref_fun(ti)', t_sim, 'UniformOutput', false));

%% ============================================================
%  7. SIMULATE CLOSED-LOOP DYNAMICS
% ============================================================
% IMPORTANT: The ODE is written in deviation coordinates, i.e. for x_dev = x - x_ref(t).
% We keep the original variable names (x_true, x_nom) for consistency, but these
% trajectories should be interpreted as deviations from the reference.

% True closed-loop
Acl_true = A - B1*K1_true - B2*K2_true;
[t, x_true] = ode45(@(t,x) Acl_true*(x - x_ref_fun(t)), t_sim, x0);

% Nominal closed-loop
Acl_nom = A - B1*K1_nom - B2*K2_nom;
[~, x_nom] = ode45(@(t,x) Acl_nom*(x - x_ref_fun(t)), t_sim, x0);

%% ============================================================
%  8. BAYESIAN PARAMETER IDENTIFICATION SETUP
% ============================================================
% Parameter vector layout (explicit):
%   Player 1: theta_1 = [ vech(P1) ; diag(Q1) ; diag(R11_free) ]
%   Player 2: theta_2 = [ vech(P2) ; diag(Q2) ; diag(R22_free) ]
%
% Here Rii_free denotes the unknown diagonal entries excluding the known scalar Rii(1,1).
% With 2x2 diagonal R matrices, this means estimating only Rii(2,2).

% True parameter vectors
true_V1 = vech(P1_true);
true_V2 = vech(P2_true);
true_params_p1 = [true_V1; diag(Q1); diag(R11(2:end, 2:end))];
true_params_p2 = [true_V2; diag(Q2); diag(R22(2:end, 2:end))];

% Prior mean (from Monte Carlo for vech(P) + analytic means for Q,R)
m01 = [V_mc_mean_p1; diag(Q1_mean); diag(R11_mean(2:end, 2:end))];
m02 = [V_mc_mean_p2; diag(Q2_mean); diag(R22_mean(2:end, 2:end))];

% Prior covariance (block diagonal)
S01 = blkdiag(V_mc_cov_p1, diag(Q1_sig.^2), diag(R11_sig(2:end).^2));
S02 = blkdiag(V_mc_cov_p2, diag(Q2_sig.^2), diag(R22_sig(2:end).^2));

% Observation noise covariance for y (must match computeOutput(...) output dimension)
Sigma2 = diag([1, 0.2, 0.2]);

% Initialize posterior
m_p1 = m01;  S_p1 = S01;
m_p2 = m02;  S_p2 = S02;

%% ============================================================
%  9. ONLINE BAYESIAN REGRESSION (objective #1)
% ============================================================
% At each time step:
%   - build regressors from (x_dev, u1, u2, x_dot)
%   - observe a noisy measurement vector y (dimension 3)
%   - update p(theta_i | data) for each player

n_steps = length(t);

theta_evolution_player1 = zeros(n_steps+1, length(m01));
theta_evolution_player2 = zeros(n_steps+1, length(m02));
var_evolution_player1   = zeros(n_steps+1, length(m01));
var_evolution_player2   = zeros(n_steps+1, length(m02));
full_covar_evolution_player1 = zeros(n_steps+1, length(m01), length(m01));
full_covar_evolution_player2 = zeros(n_steps+1, length(m02), length(m02));

% Store initial conditions
theta_evolution_player1(1,:) = m01';
theta_evolution_player2(1,:) = m02';
var_evolution_player1(1,:)   = diag(S01)';
var_evolution_player2(1,:)   = diag(S02)';
full_covar_evolution_player1(1,:,:) = S01;
full_covar_evolution_player2(1,:,:) = S02;

for k = 1:n_steps
    % Current deviation from reference
    x_t = (x_true(k,:) - x_ref_t(k,:))';

    % True control inputs (generated by true feedback)
    u1 = -K1_true * x_t;
    u2 = -K2_true * x_t;

    % State derivative (in deviation coordinates)
    x_dot = A*x_t + B1*u1 + B2*u2;

    % Compute features and regressors (helper functions in /core or /utils)
    feats   = computePlayerFeatures(x_t, {u1, u2});
    regr_p1 = computeRegressor_no_mixed(feats, x_dot, B1, 2, u1, 1);
    regr_p2 = computeRegressor_no_mixed(feats, x_dot, B2, 2, u2, 2);

    % Noisy observations:
    %   computeOutput(...) is assumed to return a 3x1 vector; Sigma2 is 3x3.
    y_p1 = computeOutput(R11(1,1), u1) + mvnrnd(zeros(1,3), Sigma2)';
    y_p2 = computeOutput(R22(1,1), u2) + mvnrnd(zeros(1,3), Sigma2)';

    % Bayesian update (posterior mean + covariance)
    [m_p1, S_p1, w_est_p1] = online_bayes_update(m_p1, S_p1, regr_p1, y_p1, Sigma2);
    [m_p2, S_p2, w_est_p2] = online_bayes_update(m_p2, S_p2, regr_p2, y_p2, Sigma2);

    % Store evolution
    theta_evolution_player1(k+1,:) = w_est_p1;
    theta_evolution_player2(k+1,:) = w_est_p2;
    var_evolution_player1(k+1,:)   = diag(S_p1)';
    var_evolution_player2(k+1,:)   = diag(S_p2)';
    full_covar_evolution_player1(k+1,:,:) = S_p1;
    full_covar_evolution_player2(k+1,:,:) = S_p2;
end

%% ============================================================
%  10. DISPLAY RESULTS
% ============================================================

display_summary(A, B1, B2, Q1_mu, Q1_sig, Q2_mu, Q2_sig, R11_mu, R11_sig, ...
                R22_mu, R22_sig, Q1, Q2, R11, R22, cv_Q, cv_Rd, eps_floor, ...
                x0, T, deltaT, V_true_p1, V_nom_p1, V_true_p2, V_nom_p2, ...
                V_mc_mean_p1, V_mc_mean_p2, valid_count, Nmc, ...
                K1_true, K1_nom, K2_true, K2_nom);

%% ============================================================
%  11. VISUALIZATION (deterministic order)
% ============================================================
% The figure order is fixed by the ordering of calls below.
% If you want deterministic filenames for paper figures, consider adding:
%   exportgraphics(gcf, fullfile('figures','Fig01_states.pdf'), 'ContentType','vector');

n_steps = size(theta_evolution_player1,1);

% Dimensions of parameter blocks
nV = n_x*(n_x+1)/2;   % length(vech(P))
nQ = n_x;             % length(diag(Q))

% --- Player 1 parameter indices ---
idx_V    = 1:nV;
idx_Q    = nV + (1:nQ);
idx_R11  = nV + nQ + (1:(n_u1-1));   % estimates diag entries excluding R11(1,1)

% --- Player 2 parameter indices ---
idx_V2   = 1:nV;
idx_Q2   = nV + (1:nQ);
idx_R22  = nV + nQ + (1:(n_u2-1));   % estimates diag entries excluding R22(1,1)

% Fig 1: state trajectories (deviation coordinates, with reference passed in)
plot_state_trajectories(t, x_true, x_nom, x_ref_fun);

% Fig 2: control inputs
plot_control_inputs(t, x_true, x_nom, K1_true, K1_nom, K2_true, K2_nom, x_ref_fun);

%% ============================================================
%  12. ONLINE PARAMETER ESTIMATION PLOTS
% ============================================================

% Fig 3-4: parameter learning curves (posterior mean/variance vs truth)
plot_parameter_learning(theta_evolution_player1, var_evolution_player1, ...
                        true_params_p1, theta_evolution_player2, ...
                        var_evolution_player2, true_params_p2, ...
                        n_u1, n_u2, n_x);

%% ============================================================
%  13. RECONSTRUCTION USING FINAL (ESTIMATED) FEATURES
% ============================================================

% Fig 5-6: reconstruction using final estimated parameters
reconstruct_and_plot(theta_evolution_player1, theta_evolution_player2, ...
                     idx_V, idx_V2, idx_R11, idx_R22, n_x, ...
                     R11, R22, B1, B2, A, x_ref_fun, t, x0, ...
                     x_true, x_nom, K1_true, K1_nom, K2_true, K2_nom);

%% ============================================================
%  14. BEHAVIOUR PREDICTION (objective #2)
% ============================================================
% Use a posterior snapshot (k=30 here) for player 1 and predict future state/input.
% This is intended to produce a predictive distribution (mean/cov + samples).

rng(4);  % keep original behaviour prediction seed for full reproducibility

k_snapshot = 30;
m_p1 = theta_evolution_player1(k_snapshot,:).';
S_p1 = squeeze(full_covar_evolution_player1(k_snapshot,:,:));

t_pred = 0:0.1:5.99;
Nmc_pred = 10000;

% Known scalar term (not estimated)
R11_known = R11(1,1);

fprintf('[Behaviour prediction] Posterior snapshot index k=%d (t=%.4f s), MC samples=%d\n', ...
        k_snapshot, t(k_snapshot), Nmc_pred);

% Fig 7-8: future predictions for state and player input trajectories
[x_mean, x_cov, u1_mean, u1_cov, x_samples, u1_samples, eps_bar] = ...
    predict_and_plot_player1_MC(x0, t_pred, m_p1, S_p1, A, B1, B2, K2_true, ...
                                idx_V, idx_R11, R11_known, x_ref_fun, Nmc_pred, x_true, K1_true);
