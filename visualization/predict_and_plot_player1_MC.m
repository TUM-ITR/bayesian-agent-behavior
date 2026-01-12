function [x_mean, x_cov, u1_mean, u1_cov, x_samples, u1_samples, eps_bar] = ...
    predict_and_plot_player1_MC(x0, t_pred, m_p1, S_p1, A, B1, B2, K2_true, ...
                                idx_V, idx_R11, R11_known, x_ref_fun, Nmc, x_true, K1_true)
% predict_and_plot_player1_MC  Behaviour prediction for player 1 via Monte Carlo.
%
%   [x_mean, x_cov, u1_mean, u1_cov, x_samples, u1_samples, eps_bar] = ...
%       predict_and_plot_player1_MC(x0, t_pred, m_p1, S_p1, A, B1, B2, K2_true, ...
%                                   idx_V, idx_R11, R11_known, x_ref_fun, Nmc, x_true, K1_true)
%
% Draws samples of player-1 parameters theta_1 from a Gaussian posterior
% theta_1 ~ N(m_p1, S_p1), reconstructs corresponding (P1, R11) and the
% induced feedback gain K1, then simulates the closed-loop dynamics forward:
%
%   x_dot = (A - B1*K1 - B2*K2_true) * (x - x_ref(t))
%
% where K2_true is kept fixed (opponent behaviour assumed known / fixed).
%
% For each Monte Carlo draw, this function returns samples of the predicted
% state trajectory x(t) and player-1 input u1(t). It also computes empirical
% mean and covariance over the Monte Carlo ensemble, and visualizes 95% bands.
%
% INPUTS:
%   x0        : (n_x x 1) initial state (at the start of prediction horizon)
%   t_pred    : (1 x H) or (H x 1) vector of prediction times passed to ode45
%               (Interpretation of absolute vs relative time depends on x_ref_fun.)
%
%   m_p1      : (d1 x 1) posterior mean of player-1 parameter vector theta_1
%   S_p1      : (d1 x d1) posterior covariance of theta_1
%
%   A         : (n_x x n_x) system matrix
%   B1, B2    : (n_x x n_u1), (n_x x n_u2) input matrices
%   K2_true   : (n_u2 x n_x) fixed gain used for player 2 in prediction
%
%   idx_V     : indices selecting vech(P1) inside theta_1
%   idx_R11   : indices selecting the estimated diagonal entries of R11 excluding R11(1,1)
%   R11_known : scalar, known entry R11(1,1) (assumed fixed / not estimated)
%
%   x_ref_fun : function handle returning reference x_ref(t) (n_x x 1)
%   Nmc       : number of Monte Carlo samples
%
%   x_true    : (Nsim x n_x) state trajectory from simulation (used only for optional overlay in plots)
%   K1_true   : (n_u1 x n_x) true gain (used only for optional overlay in plots)
%
% OUTPUTS:
%   x_mean    : (n_x x H) Monte Carlo mean of predicted state
%   x_cov     : (n_x x n_x x H) Monte Carlo covariance of predicted state
%   u1_mean   : (n_u1 x H) Monte Carlo mean of predicted input u1
%   u1_cov    : (n_u1 x n_u1 x H) Monte Carlo covariance of predicted input u1
%   x_samples : (n_x x H x Nmc) sample trajectories for x
%   u1_samples: (n_u1 x H x Nmc) sample trajectories for u1
%   eps_bar   : scenario-theoretic bound (see scenario_epsilon_star)
%
% NOTES / ASSUMPTIONS:
%   - Only player-1 uncertainty is propagated (player 2 is fixed at K2_true).
%   - The prediction dynamics are expressed in deviation form (x - x_ref(t)).
%   - The "true trajectory overlay" below is currently a coarse subsample of x_true;
%     it is intended as a qualitative comparison only.
%   - The scenario bound uses a heuristic choice of decision dimension d (see below).
%

    % ----------------------------------------------
    % Monte Carlo prediction for Player 1 with ODE45
    % ----------------------------------------------
    n_x  = size(A,1);
    n_u1 = size(B1,2);
    H    = length(t_pred);

    x_samples  = zeros(n_x,  H, Nmc);
    u1_samples = zeros(n_u1, H, Nmc);

    % Cholesky factor of posterior covariance
    L1 = chol(S_p1, 'lower');

    for s = 1:Nmc
        % Sample theta_1 from Gaussian posterior
        eps_s = randn(length(m_p1),1);
        theta1_s = m_p1 + L1*eps_s;

        % Reconstruct P1 and diagonal R11 from sampled parameters
        P1_s = ivech_upper(theta1_s(idx_V), n_x);

        R11_tail_s = theta1_s(idx_R11);      % estimated diagonal entries excluding the known first one
        R11_diag_s = [R11_known; R11_tail_s];
        R11_s = diag(R11_diag_s);

        % Player 1 gain from (P1, R11)
        K1_s = R11_s \ (B1' * P1_s);

        % Closed-loop simulation over the prediction horizon
        odefun = @(t,x) (A - B1*K1_s - B2*K2_true) * (x - x_ref_fun(t));
        [t_s, x_s] = ode45(odefun, t_pred, x0);

        % Store states in shape (n_x x H)
        x_s = x_s.';
        x_samples(:,:,s) = x_s;

        % Compute player-1 inputs along the predicted trajectory
        for k = 1:H
            xk    = x_s(:,k);
            xrefk = x_ref_fun(t_s(k));
            u1_samples(:,k,s) = -K1_s * (xk - xrefk);
        end
    end

    % --- Empirical mean & covariance over Monte Carlo ensemble ---
    x_mean  = mean(x_samples,3);
    u1_mean = mean(u1_samples,3);

    x_cov  = zeros(n_x,n_x,H);
    u1_cov = zeros(n_u1,n_u1,H);

    for k = 1:H
        Xk  = squeeze(x_samples(:,k,:)).';   % (Nmc x n_x)
        U1k = squeeze(u1_samples(:,k,:)).';  % (Nmc x n_u1)
        x_cov(:,:,k)  = cov(Xk);
        u1_cov(:,:,k) = cov(U1k);
    end

    % --- Plot predicted states and inputs with 95% CI ---
    % NOTE: The "true" overlay below is a coarse subsample for qualitative comparison.
    % If you have a matching time grid for x_true, replace this with a proper alignment.
    x_true_plot  = x_true(1:10:600,:)';          % optional: replace with properly aligned truth
    u1_true_plot = -K1_true * x_true_plot;       % optional

    plot_predicted_state_input_95CI(t_pred, x_samples, u1_samples, x_true_plot, u1_true_plot, x_ref_fun);

    % --- Scenario-theoretic certified violation probability (heuristic d) ---
    % IMPORTANT: The choice of "d" below is problem-dependent. Here it is used
    % as a conservative illustrative dimension count for an envelope constraint.
    d = 2*H*n_x;        % example decision dimension (document/adjust as needed)
    beta = 1e-2;        % confidence level (probability of failure of the certificate)
    eps_bar = scenario_epsilon_star(Nmc, d, beta);

    fprintf('Certified violation probability (scenario bound): %.4g\n', eps_bar);
end

%% ---------------- Scenario-theory helper ----------------
function eps_bar = scenario_epsilon_star(Nmc, d, beta)
% scenario_epsilon_star  Compute scenario bound epsilon* for given (Nmc, d, beta).
%
% Solves for eps_bar in:
%   sum_{i=0}^{d-1} C(Nmc,i) eps^i (1-eps)^(Nmc-i) = beta
%
% This is one common form of a scenario-theory tail bound. The interpretation
% depends on how "d" is chosen (support rank / number of effective decision vars).

    logBinom = @(n,k) gammaln(n+1) - gammaln(k+1) - gammaln(n-k+1);
    tail = @(eps) sum(arrayfun(@(i) exp(logBinom(Nmc,i) + ...
                          i*log(eps) + (Nmc-i)*log(1-eps)), 0:(d-1)));
    f = @(eps) tail(eps) - beta;

    lb = 1e-12; ub = 1-1e-12;
    if f(lb) < 0, eps_bar = lb; return; end
    if f(ub) > 0, eps_bar = ub; return; end
    eps_bar = fzero(f, [lb ub]);
end

%% ---------------- Plot 95% CI & Envelope ----------------
function plot_predicted_state_input_95CI(t_pred, x_samples, u1_samples, x_true, u1_true, x_ref_fun)
% plot_predicted_state_input_95CI  Plot mean, 95% CI, and envelope for predicted trajectories.
%
% Plots (for each state/input component):
%   - 95% pointwise interval (2.5th–97.5th percentile across samples)
%   - min/max envelope across samples
%   - sample mean
%   - optional "true" overlay and the reference (for states)

    fs = 20; lw_mean = 1.5; lw_true = 2.5;
    [n_x,~,~]  = size(x_samples);
    [n_u1,~,~] = size(u1_samples);
    colors = lines(max(n_x,n_u1));

    % Reference trajectory on prediction grid
    x_ref_mat = cell2mat(arrayfun(@(ti) x_ref_fun(ti), t_pred, 'UniformOutput', false));

    % ----------- States -----------
    figure('Name','State Prediction + 95% CI & Envelope');
    tiledlayout(n_x,1,'TileSpacing','none','Padding','none');

    for i = 1:n_x
        ax = nexttile;
        hold(ax,'on'); grid(ax,'on'); box(ax,'on');

        Xi = squeeze(x_samples(i,:,:)).';   % (Nmc x H)
        mu = mean(Xi,1);
        lb = prctile(Xi,2.5,1);
        ub = prctile(Xi,97.5,1);
        env_lb = min(Xi,[],1);
        env_ub = max(Xi,[],1);

        fill([t_pred, fliplr(t_pred)], [ub, fliplr(lb)], colors(i,:), 'FaceAlpha',0.15,'EdgeColor','none');
        plot(t_pred, env_ub, '--', 'Color', colors(i,:), 'LineWidth', 1.5);
        plot(t_pred, env_lb, '--', 'Color', colors(i,:), 'LineWidth', 1.5);
        plot(t_pred, mu,     'Color', colors(i,:), 'LineWidth', lw_mean);

        plot(t_pred, x_true(i,:),    'k-', 'LineWidth', lw_true);
        plot(t_pred, x_ref_mat(i,:), 'k',  'LineWidth', 1);

        ylabel(sprintf('$x_{%d}$',i),'Interpreter','latex','FontSize',fs);
        text(0.98,0.99,sprintf('$x_{%d}$',i),'Units','normalized','Interpreter','latex', ...
             'HorizontalAlignment','right','VerticalAlignment','top','FontSize',fs);

        if i==n_x
            xlabel('Time [s]','Interpreter','latex','FontSize',fs);
        else
            set(ax,'XTickLabel',[]);
        end

        set(ax,'FontSize',fs,'TickLabelInterpreter','latex');
    end

    % ----------- Inputs -----------
    figure('Name','Player 1 Input Prediction + 95% CI & Envelope');
    tiledlayout(n_u1,1,'TileSpacing','none','Padding','none');

    for k = 1:n_u1
        ax = nexttile;
        hold(ax,'on'); grid(ax,'on'); box(ax,'on');

        Uk = squeeze(u1_samples(k,:,:)).';   % (Nmc x H)
        mu_u = mean(Uk,1);
        lb_u = prctile(Uk,2.5,1);
        ub_u = prctile(Uk,97.5,1);
        env_lb_u = min(Uk,[],1);
        env_ub_u = max(Uk,[],1);

        fill([t_pred, fliplr(t_pred)], [ub_u, fliplr(lb_u)], colors(k,:), 'FaceAlpha',0.15,'EdgeColor','none');
        plot(t_pred, env_ub_u, '--', 'Color', colors(k,:), 'LineWidth', 1.5);
        plot(t_pred, env_lb_u, '--', 'Color', colors(k,:), 'LineWidth', 1.5);
        plot(t_pred, mu_u,     'Color', colors(k,:), 'LineWidth', lw_mean);

        plot(t_pred, u1_true(k,:), 'k-', 'LineWidth', lw_true);

        ylabel(sprintf('$u_{1,%d}$',k),'Interpreter','latex','FontSize',fs);
        text(0.98,0.2,sprintf('$u_{1,%d}$',k),'Units','normalized','Interpreter','latex', ...
             'HorizontalAlignment','right','VerticalAlignment','top','FontSize',fs);

        if k==n_u1
            xlabel('Time [s]','Interpreter','latex','FontSize',fs);
        else
            set(ax,'XTickLabel',[]);
        end

        set(ax,'FontSize',fs,'TickLabelInterpreter','latex');
    end
end
