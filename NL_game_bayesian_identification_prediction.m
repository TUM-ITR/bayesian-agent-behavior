%% ============================================================
% BAYESIAN PARAMETER IDENTIFICATION & BEHAVIOUR PREDICTION
% in a 1D NONLINEAR DIFFERENTIAL GAME (PAPER EXAMPLE)
% ============================================================
% This script reproduces a 1D nonlinear two-player differential game example
% used in the paper. It considers nonlinear scalar dynamics of the form:
%
%     x_dot = f(x) + g1(x) u1 + g2(x) u2
%
% and focuses on the same two main objectives as the LQ example:
%
%   (1) Online Bayesian parameter estimation
%       - Gaussian priors on (scalar) state-cost parameters Q1, Q2
%       - Recursive Bayesian regression on a stream of noisy observations
%       - Estimation of a truncated set of value-function parameters (weights)
%         together with the scalar Q_i
%
%   (2) Behaviour prediction and posterior analysis
%       - Generate a closed-loop trajectory using a reference ("true") solution
%         obtained from a high-order basis (ground truth)
%       - Visualize the learned parameters and their uncertainty (mean ± 2σ)
%       - Reconstruct value functions on a state grid using the posterior
%         covariance (mean ± 2σ bands)
%
% Notes for readers:
%   - This file assumes the existence of the helper functions:
%       compute_value_functions_dyn, computeRegressor_no_mixed,
%       computeOutput, online_bayes_update
% ============================================================

clear; clc; close all;
rng(1); % reproducibility (global seed)

%% ============================================================
%  1. NONLINEAR DYNAMICS SETUP
% ============================================================

b0 = 2.0;
beta = 0.3;
b = @(x) b0*(1 + beta*x.^2);

% Baseline example (paper)
f  = @(x) (-0.5*x - 0.3*x.^3)./b(x);
g1 = @(x) (1.2 + 0.8*sin(1.5*x))./b(x);              % oscillating human influence
g2 = @(x) (1.0 - 0.7*sin(1.5*x + pi/3))./b(x);       % roughly out-of-phase robot

R1 = 1;
R2 = 1;

%% ============================================================
%  2. PRIORS AND TRUE Q SAMPLING
% ============================================================

% Prior means (scalar state-cost weights)
Q1_mean = 1.0;
Q2_mean = 0.3;

% Coefficient of variation (relative std)
cv_Q = 0.5;
eps_floor = 1e-3;

% Truncated Gaussian sampling (scalar case)
sample_diag_gaussian = @(mu,cv) max(mu + cv*mu*randn(), eps_floor);

% Sample true costs
Q1_true = sample_diag_gaussian(Q1_mean, cv_Q);
Q2_true = sample_diag_gaussian(Q2_mean, cv_Q);

fprintf('True Q1=%.3f, Q2=%.3f (priors mean %.2f, %.2f, CV=%.2f)\n', ...
    Q1_true, Q2_true, Q1_mean, Q2_mean, cv_Q);

%% ============================================================
%  3. COMPUTE TRUE VALUE FUNCTIONS (HIGH-ORDER BASIS)
% ============================================================

alg_true.L = 5;     % domain half-width (state grid typically in [-L, L])
alg_true.M = 201;   % grid points / discretization (depends on implementation)
alg_true.K = 10;    % number of basis functions
alg_true.g1 = g1;
alg_true.g2 = g2;

[V1_true_fun, V2_true_fun, res_true] = ...
    compute_value_functions_dyn(f, Q1_true, Q2_true, R1, R2, alg_true);

%% ============================================================
%  3b. VISUALIZE VALUE-FUNCTION BASIS FEATURES
% ============================================================

xgrid = linspace(-alg_true.L, alg_true.L, 400);

% Evaluate basis over x-grid. Phi_fun is assumed to return basis evaluations.
Phi = res_true.Phi_fun(xgrid');   % returns basis values for each x
Phi = squeeze(Phi);

% Orientation handling for plotting (expect Phi_plot(i,:) corresponds to feature i)
Phi_plot = Phi';

figure('Name','Value-function basis features','Color','w');
tiledlayout('flow','TileSpacing','compact','Padding','compact');

for i = 1:alg_true.K
    nexttile;
    plot(xgrid, Phi_plot(i,:), 'LineWidth', 1.5);
    title(sprintf('\\Phi_{%d}(x)', i), 'Interpreter', 'tex');
    xlabel('x');
    ylabel(sprintf('Feature %d', i));
    grid on;
end

sgtitle(sprintf('Basis functions in res\\_true.Phi\\_fun (K = %d)', alg_true.K));

%% ============================================================
%  4. GENERATE A CLOSED-LOOP TRAJECTORY (ODE45)
% ============================================================

dt = 0.01;
T1 = 4;         % first segment duration
T2 = 8;         % final time after switching IC
x0_1 = 4.0;     % IC for first segment

odefun = @(t,x) closed_loop_dyn(x, f, g1, g2, R1, R2, res_true);

opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-9);

% --- First segment: 0 -> T1 ---
[t1, x1] = ode45(odefun, 0:dt:T1, x0_1, opts);

% --- Switch initial condition at t = T1 ---
x0_2 = -abs(x0_1);   % example switch: same magnitude but negative sign

% --- Second segment: T1 -> T2 ---
[t2, x2] = ode45(odefun, T1:dt:T2, x0_2, opts);

% --- Concatenate ---
tgrid = [t1; t2];
x     = [x1; x2];

% Preallocate diagnostics / trajectories
u1_traj   = zeros(length(x), 1);
u2_traj   = zeros(length(x), 1);
xdot_traj = zeros(length(x), 1);

% Reconstruct controls and derivatives along the generated trajectory
for k = 1:length(x)
    xi = x(k);

    [~, dPhi] = res_true.Phi_fun(xi);
    V1x = dPhi * res_true.w1;
    V2x = dPhi * res_true.w2;

    % NOTE: scaling factors depend on paper's feature/cost normalization.
    u1 = -(g1(xi)/(2*R1)) * V1x;
    u2 = -(g2(xi)/(2*R2)) * V2x;

    xdot = f(xi) + g1(xi)*u1 + g2(xi)*u2;

    u1_traj(k) = u1;
    u2_traj(k) = u2;
    xdot_traj(k) = xdot;
end

%% ============================================================
%  4b. PLOT TRAJECTORY AND CONTROL INPUTS
% ============================================================

figure('Name','State trajectory and control inputs (ode45)','Color','w');
tiledlayout(3,1,'TileSpacing','compact','Padding','compact');

nexttile; hold on; grid on;
plot(tgrid, x, 'LineWidth', 1.6);
xlabel('Time [s]', 'Interpreter', 'latex');
ylabel('$x(t)$', 'Interpreter', 'latex');
title('State trajectory', 'Interpreter', 'latex');

nexttile; hold on; grid on;
plot(tgrid, u1_traj, 'LineWidth', 1.4);
xlabel('Time [s]', 'Interpreter', 'latex');
ylabel('$u_1^*(t)$', 'Interpreter', 'latex');
title('Player 1 control', 'Interpreter', 'latex');

nexttile; hold on; grid on;
plot(tgrid, u2_traj, 'LineWidth', 1.4);
xlabel('Time [s]', 'Interpreter', 'latex');
ylabel('$u_2^*(t)$', 'Interpreter', 'latex');
title('Player 2 control', 'Interpreter', 'latex');

%% ============================================================
%  5. BAYESIAN ESTIMATOR INITIALIZATION
% ============================================================

n_steps = length(tgrid);

% Estimator uses a truncated feature set (subset of the true basis)
K_est = alg_true.K;          % number of basis functions used by estimator

% Parameter vector size:
%   theta_i = [ w_i ((K_est-2) x 1) ; Q_i (scalar) ]
p_i   = K_est - 1;

% Measurement noise covariance (2 equations per step in this setup)
Sigma_y = 0.1 * diag([0.5, 0.5]);

% Prior means used by the estimator (may differ from sampling means)
Q1_prior_mean = 1.0;
Q2_prior_mean = 0.6;

% Prior covariance (simple diagonal prior on parameters)
S01 = blkdiag(eye(K_est-2), (cv_Q * Q1_prior_mean)^2);
S02 = blkdiag(eye(K_est-2), (cv_Q * Q2_prior_mean)^2);

m01 = [zeros(K_est-2, 1); Q1_prior_mean];
m02 = [zeros(K_est-2, 1); Q2_prior_mean];

% Allocate histories (means, variances, full covariance)
theta_evolution_player1 = zeros(n_steps+1, p_i);
theta_evolution_player2 = zeros(n_steps+1, p_i);
var_evolution_player1   = zeros(n_steps+1, p_i);
var_evolution_player2   = zeros(n_steps+1, p_i);
full_covar_evolution_player1 = zeros(n_steps+1, p_i, p_i);
full_covar_evolution_player2 = zeros(n_steps+1, p_i, p_i);

% Initialize posteriors with priors (scaled here as in the original script)
m_p1 = m01; S_p1 = 2*S01;
m_p2 = m02; S_p2 = 2*S02;

theta_evolution_player1(1,:) = m_p1';
theta_evolution_player2(1,:) = m_p2';
var_evolution_player1(1,:)   = diag(S_p1)';
var_evolution_player2(1,:)   = diag(S_p2)';

% FIX: correct storage at initialization
full_covar_evolution_player1(1,:,:) = S_p1;
full_covar_evolution_player2(1,:,:) = S_p2;

%% ============================================================
%  6. ONLINE BAYESIAN UPDATE LOOP
% ============================================================

for k = 1:n_steps

    xk    = x(k);
    uk1   = u1_traj(k);
    uk2   = u2_traj(k);
    xdotk = xdot_traj(k);

    % --- Feature construction (match estimator feature set) ---
    [Phi_est, dPhi_est] = res_true.Phi_fun(xk);

    % Truncate to estimator feature set
    Phi_est  = Phi_est(1:K_est);
    dPhi_est = dPhi_est(1:K_est);

    % Value-function features only (exclude first two basis terms by design)
    feats.sigmaVi     = Phi_est(3:end)';      % (K_est-2) x 1
    feats.gradSigmaVi = dPhi_est(3:end)';     % (K_est-2) x 1 (1D gradient)
    feats.sigmaQi     = xk^2;                 % scalar state-cost feature
    feats.sigmaRi     = [uk1^2, uk2^2]';      % input-cost features (no cross)

    % --- Build linear regressors ---
    % g1(xk), g2(xk) are scalars (1×1 matrices) in this 1D example.
    H1 = computeRegressor_no_mixed(feats, xdotk, g1(xk), 2, uk1, 1);
    H2 = computeRegressor_no_mixed(feats, xdotk, g2(xk), 2, uk2, 2);

    % --- Synthetic measurements ---
    y1 = computeOutput(R1, uk1) + mvnrnd([0 0], Sigma_y)';
    y2 = computeOutput(R2, uk2) + mvnrnd([0 0], Sigma_y)';

    % --- Recursive Bayesian updates ---
    [m_p1, S_p1] = online_bayes_update(m_p1, S_p1, H1, y1, Sigma_y);
    [m_p2, S_p2] = online_bayes_update(m_p2, S_p2, H2, y2, Sigma_y);

    % --- Store results ---
    theta_evolution_player1(k+1,:) = m_p1';
    theta_evolution_player2(k+1,:) = m_p2';
    var_evolution_player1(k+1,:)   = diag(S_p1)';
    var_evolution_player2(k+1,:)   = diag(S_p2)';
    full_covar_evolution_player1(k+1,:,:) = S_p1;
    full_covar_evolution_player2(k+1,:,:) = S_p2;
end

fprintf('Final posterior means: Q1 = %.3f, Q2 = %.3f\n', ...
        m_p1(end), m_p2(end));

%% ============================================================
%  7. POSTERIOR ANALYSIS AND RELATIVE ERROR PLOTS
% ============================================================

% True parameters (consistent with estimator truncation)
true_w1 = res_true.w1(3:K_est);
true_w2 = res_true.w2(3:K_est);
true_Q1 = Q1_true;
true_Q2 = Q2_true;

% Extract posterior means/variances
w1_mean = theta_evolution_player1(:,1:end-1);
w2_mean = theta_evolution_player2(:,1:end-1);
Q1_est_mean_traj = theta_evolution_player1(:,end);
Q2_est_mean_traj = theta_evolution_player2(:,end);

w1_var  = var_evolution_player1(:,1:end-1);
w2_var  = var_evolution_player2(:,1:end-1);

% Relative errors
rel_err_w1 = vecnorm(w1_mean - true_w1', 2, 2) ./ (norm(true_w1) + eps);
rel_err_w2 = vecnorm(w2_mean - true_w2', 2, 2) ./ (norm(true_w2) + eps);
rel_err_Q1 = abs(Q1_est_mean_traj - true_Q1) ./ (abs(true_Q1) + eps);
rel_err_Q2 = abs(Q2_est_mean_traj - true_Q2) ./ (abs(true_Q2) + eps);

fprintf('\n=== Final Relative Errors ===\n');
fprintf('||w1_est - w1_true|| / ||w1_true|| = %.3f\n', rel_err_w1(end));
fprintf('||w2_est - w2_true|| / ||w2_true|| = %.3f\n', rel_err_w2(end));
fprintf('|Q1_est - Q1_true| / |Q1_true| = %.3f\n', rel_err_Q1(end));
fprintf('|Q2_est - Q2_true| / |Q2_true| = %.3f\n', rel_err_Q2(end));

%% ============================================================
%  8. PLOTS: PARAMETER LEARNING
% ============================================================

colors = lines(max(K_est,3));

% --- (1) Feature weights evolution for player 1 ---
figure('Name','Player 1 Feature Weights Posterior','Color','w');
hold on; grid on;

for i = 1:K_est-2
    mu    = w1_mean(:,i);
    sigma = sqrt(w1_var(:,i));

    fill([1:n_steps+1, fliplr(1:n_steps+1)], ...
         [mu'+2*sigma', fliplr(mu'-2*sigma')], ...
         colors(i,:), 'FaceAlpha',0.15,'EdgeColor','none');

    plot(mu, 'LineWidth',1.6,'Color',colors(i,:));
    yline(true_w1(i),'--','Color',colors(i,:),'LineWidth',1.4);
end

xlabel('Time step','Interpreter','latex');
ylabel('$w_{1,i}$','Interpreter','latex');
title('Player 1: Posterior Feature Weights (mean $\pm 2\sigma$)','Interpreter','latex');

h_band = fill(NaN,NaN,'k','FaceAlpha',0.15,'EdgeColor','none');
h_true = plot(NaN,NaN,'--k','LineWidth',1.4);
h_mean = plot(NaN,NaN,'-k','LineWidth',1.6);
legend([h_mean h_true h_band], {'Posterior mean','True','\pm 2\sigma band'}, ...
       'Interpreter','latex','Location','bestoutside');

% --- (2) Feature weights evolution for player 2 ---
figure('Name','Player 2 Feature Weights Posterior','Color','w');
hold on; grid on;

for i = 1:K_est-2
    mu    = w2_mean(:,i);
    sigma = sqrt(w2_var(:,i));

    fill([1:n_steps+1, fliplr(1:n_steps+1)], ...
         [mu'+2*sigma', fliplr(mu'-2*sigma')], ...
         colors(i,:), 'FaceAlpha',0.15,'EdgeColor','none');

    plot(mu, 'LineWidth',1.6,'Color',colors(i,:));
    yline(true_w2(i),'--','Color',colors(i,:),'LineWidth',1.4);
end

xlabel('Time step','Interpreter','latex');
ylabel('$w_{2,i}$','Interpreter','latex');
title('Player 2: Posterior Feature Weights (mean $\pm 2\sigma$)','Interpreter','latex');

h_band = fill(NaN,NaN,'k','FaceAlpha',0.15,'EdgeColor','none');
h_true = plot(NaN,NaN,'--k','LineWidth',1.4);
h_mean = plot(NaN,NaN,'-k','LineWidth',1.6);
legend([h_mean h_true h_band], {'Posterior mean','True','\pm 2\sigma band'}, ...
       'Interpreter','latex','Location','bestoutside');

% --- (3) Q parameter evolution ---
figure('Name','Posterior Means for Q1,Q2','Color','w');

subplot(2,1,1); hold on; grid on;
plot(Q1_est_mean_traj, 'b','LineWidth',1.8);
yline(true_Q1,'--k','LineWidth',1.6);
xlabel('Time step'); ylabel('$Q_1$','Interpreter','latex');
title(sprintf('Player 1: $Q_1$ estimate (final rel. err = %.2f%%)', ...
               100*rel_err_Q1(end)),'Interpreter','latex');

subplot(2,1,2); hold on; grid on;
plot(Q2_est_mean_traj, 'r','LineWidth',1.8);
yline(true_Q2,'--k','LineWidth',1.6);
xlabel('Time step'); ylabel('$Q_2$','Interpreter','latex');
title(sprintf('Player 2: $Q_2$ estimate (final rel. err = %.2f%%)', ...
               100*rel_err_Q2(end)),'Interpreter','latex');

% --- (4) Relative error trajectories ---
figure('Name','Relative Parameter Errors','Color','w');
semilogy(rel_err_w1,'b','LineWidth',1.8); hold on; grid on;
semilogy(rel_err_w2,'r','LineWidth',1.8);
semilogy(rel_err_Q1,'--b','LineWidth',1.8);
semilogy(rel_err_Q2,'--r','LineWidth',1.8);
xlabel('Time step');
ylabel('Relative error (log scale)','Interpreter','latex');
legend({'$w_1$','$w_2$','$Q_1$','$Q_2$'},'Interpreter','latex','Location','best');
title('Convergence of Posterior Means','Interpreter','latex');

%% ============================================================
%  9. BAYESIAN VALUE-FUNCTION RECONSTRUCTION (MEAN ± 2σ)
% ============================================================

k_eval = 500;  % choose iteration (example snapshot)
xgrid_val = linspace(-alg_true.L, alg_true.L, 200);
Nx_val    = numel(xgrid_val);

V1_mean_grid = zeros(1, Nx_val);
V2_mean_grid = zeros(1, Nx_val);
V1_std_grid  = zeros(1, Nx_val);
V2_std_grid  = zeros(1, Nx_val);

% Posterior mean (only w_i, exclude Q)
w1_mean_k = theta_evolution_player1(k_eval, 1:end-1)';   % (K_est-2)x1
w2_mean_k = theta_evolution_player2(k_eval, 1:end-1)';

% Full covariance and truncate to w-block
S1_full = squeeze(full_covar_evolution_player1(k_eval, :, :));
S2_full = squeeze(full_covar_evolution_player2(k_eval, :, :));

S1_w = S1_full(1:end-1, 1:end-1);
S2_w = S2_full(1:end-1, 1:end-1);

for j = 1:Nx_val
    xj = xgrid_val(j);

    [Phi_est, ~] = res_true.Phi_fun(xj);
    Phi_est = Phi_est(1:K_est);

    sigmaVi = Phi_est(3:end);   % (K_est-2)x1

    V1_mean_grid(j) = sigmaVi * w1_mean_k;
    V1_var_j        = sigmaVi * S1_w * sigmaVi';
    V1_std_grid(j)  = sqrt(max(V1_var_j, 0));

    V2_mean_grid(j) = sigmaVi * w2_mean_k;
    V2_var_j        = sigmaVi * S2_w * sigmaVi';
    V2_std_grid(j)  = sqrt(max(V2_var_j, 0));
end

% Shift for visualization (shape comparison)
V1_mean_grid = V1_mean_grid - min(V1_mean_grid);
V2_mean_grid = V2_mean_grid - min(V2_mean_grid);

% True value functions
V1_true_grid = V1_true_fun(xgrid_val');
V2_true_grid = V2_true_fun(xgrid_val');

figure('Name', sprintf('Bayesian Value Function Estimates (k = %d)', k_eval), ...
       'Color', 'w');

tiledlayout(2,1,'TileSpacing','none','Padding','none');

fs = 20; lw = 1.6;

ax1 = nexttile; hold(ax1,'on'); grid(ax1,'on');

fill([xgrid_val, fliplr(xgrid_val)], ...
     [V1_mean_grid + 2*V1_std_grid, fliplr(V1_mean_grid - 2*V1_std_grid)], ...
     [0.8 0.8 1.0], 'FaceAlpha', 0.9, 'EdgeColor', 'none');

plot(xgrid_val, V1_true_grid, '--k', 'LineWidth', lw);
plot(xgrid_val, V1_mean_grid, 'b', 'LineWidth', lw);

set(ax1,'FontSize',fs,'TickLabelInterpreter','latex');
xlim(ax1, [min(xgrid_val), max(xgrid_val)]);
ylim(ax1, [0, 20]);
set(ax1, 'XTicklabel', []);
set(ax1, 'YTick', 0:6:18);

text(ax1,0.5,0.92,'$V_1(x)$','Units','normalized', ...
     'Interpreter','latex','FontSize',fs, ...
     'HorizontalAlignment','center');

box(ax1,'on');

ax2 = nexttile; hold(ax2,'on'); grid(ax2,'on');

fill([xgrid_val, fliplr(xgrid_val)], ...
     [V2_mean_grid + 2*V2_std_grid, fliplr(V2_mean_grid - 2*V2_std_grid)], ...
     [1.0 0.8 0.8], 'FaceAlpha', 0.9, 'EdgeColor', 'none');

plot(xgrid_val, V2_true_grid, '--k', 'LineWidth', lw);
plot(xgrid_val, V2_mean_grid, 'r', 'LineWidth', lw);

set(ax2,'FontSize',fs,'TickLabelInterpreter','latex');
xlim(ax2, [min(xgrid_val), max(xgrid_val)]);
ylim(ax2, [0, 15.5]);
set(ax2, 'YTick', 0:4:15);

text(ax2,0.5,0.92,'$V_2(x)$','Units','normalized', ...
     'Interpreter','latex','FontSize',fs, ...
     'HorizontalAlignment','center');

text(ax2,0.98,0.02,'$x$', ...
     'Units','normalized','Interpreter','latex', ...
     'HorizontalAlignment','right','VerticalAlignment','bottom', ...
     'FontSize',fs);

box(ax2,'on');

h_mean = plot(ax1, NaN, NaN, '-',  'Color', 'k', 'LineWidth', lw);
h_true = plot(ax1, NaN, NaN, '--', 'Color', 'k', 'LineWidth', lw);
h_band = fill(ax1, [0 0], [0 0], 'k', 'FaceAlpha', 0.15, 'EdgeColor', 'none');

lg = legend(ax1, [h_band h_true h_mean], ...
    {'$\pm 2\sigma$ band','True value','Posterior mean'}, ...
    'Interpreter','latex','FontSize',fs, ...
    'Orientation','horizontal','Location','northoutside');
lg.Box = 'off';

% set(gcf,'Units','pixels');
% pos = get(gcf,'Position');
% set(gcf,'PaperPositionMode','auto','PaperUnits','points');
% set(gcf,'PaperSize',[pos(3) pos(4)]);

%% ============================================================
%  LOCAL HELPER: CLOSED-LOOP DYNAMICS
% ============================================================

function dx = closed_loop_dyn(x, f, g1, g2, R1, R2, res)
% CLOSED_LOOP_DYN Closed-loop dynamics using value gradients from res.Phi_fun.
%
% Uses basis derivative to compute value gradients and applies the feedback
% policy according to the normalization used in the paper.
%
% NOTE: Scaling factors (e.g., 2*R) depend on the specific HJB normalization.

    [~, dPhi] = res.Phi_fun(x);
    V1x = dPhi * res.w1;
    V2x = dPhi * res.w2;

    u1 = -(g1(x)/(2*R1)) * V1x;
    u2 = -(g2(x)/(2*R2)) * V2x;

    dx = f(x) + g1(x)*u1 + g2(x)*u2;
end
