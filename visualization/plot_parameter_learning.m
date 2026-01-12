function plot_parameter_learning(theta_evolution_player1, var_evolution_player1, ...
                                 true_params_p1, theta_evolution_player2, ...
                                 var_evolution_player2, true_params_p2, ...
                                 n_u1, n_u2, n_x)
% plot_parameter_learning  Visualize online Bayesian learning of parameters (2 players).
%
%   plot_parameter_learning(theta_evolution_player1, var_evolution_player1, true_params_p1, ...
%                           theta_evolution_player2, var_evolution_player2, true_params_p2, ...
%                           n_u1, n_u2, n_x)
%
% Plots the evolution of posterior mean and uncertainty (±2σ band) for the
% estimated parameter vectors of both players. The parameter vector is assumed
% to be ordered as:
%
%   theta_i = [ vech(P_i) ; diag(Q_i) ; diag(R_ii_free) ]
%
% where vech(P_i) stacks the upper triangular entries of P_i (row-wise) and
% R_ii_free excludes the known scalar R_ii(1,1).
%
% INPUTS:
%   theta_evolution_player1 : (N x d1) matrix
%       Posterior mean trajectory (or stored estimate) for player 1 parameters
%   var_evolution_player1   : (N x d1) matrix
%       Posterior variances (diagonal of covariance) for player 1 parameters
%   true_params_p1          : (d1 x 1) vector
%       Ground truth parameter vector for player 1
%
%   theta_evolution_player2 : (N x d2) matrix
%       Posterior mean trajectory for player 2 parameters
%   var_evolution_player2   : (N x d2) matrix
%       Posterior variances for player 2 parameters
%   true_params_p2          : (d2 x 1) vector
%       Ground truth parameter vector for player 2
%
%   n_u1, n_u2              : scalars
%       Number of input channels for players 1 and 2
%   n_x                     : scalar
%       Number of states (used to define diag(Q) and vech(P) block sizes)
%
% OUTPUT:
%   None (creates figures).
%
% NOTES:
%   - The x-axis is the *update index* (time step index), not physical time.
%   - Axis limits/ticks are currently set for the default experiment and may be
%     adjusted depending on parameter scales.
%

    %% Setup
    n_steps = size(theta_evolution_player1,1);

    % Block sizes in the assumed parameter vector
    nV = n_x*(n_x+1)/2;   % length(vech(P))
    nQ = n_x;             % length(diag(Q))

    % --- Player 1 indices ---
    idx_V    = 1:nV;
    idx_Q    = nV + (1:nQ);
    idx_R11  = nV + nQ + (1:(n_u1-1));

    % --- Player 2 indices ---
    idx_V2   = 1:nV;
    idx_Q2   = nV + (1:nQ);
    idx_R22  = nV + nQ + (1:(n_u2-1));

    % --- Color palette ---
    max_params = max([nV, nQ, n_u1, n_u2]);
    colors = lines(max_params);

    fs = 20;  % font size
    lw = 1.5; % line width

    %% ---------------- PLAYER 1 ----------------
    figure('Name','Player 1 Parameter Learning');
    tlo = tiledlayout(3,1,'TileSpacing','none','Padding','none');

    % (y-lims/ticks are tuned for the default setup; adjust if needed)
    plot_params_subplot(tlo, idx_V,   theta_evolution_player1, var_evolution_player1, true_params_p1, colors, lw, fs, 'V_1', [-2,12],  0:4:11, []);
    plot_params_subplot(tlo, idx_Q,   theta_evolution_player1, var_evolution_player1, true_params_p1, colors, lw, fs, 'Q_1', [-3,13],  0:5:10, []);
    plot_params_subplot(tlo, idx_R11, theta_evolution_player1, var_evolution_player1, true_params_p1, colors, lw, fs, 'R_1', [0.2,2.8], 0.5:1:2.5, 'Update index');

    % Optional: export for paper figure (kept as in original code; adjust filename/path as desired)
    set(gcf, 'PaperPositionMode', 'auto');
    print(gcf, '-dsvg', 'params_p1.svg');

    %% ---------------- PLAYER 2 ----------------
    figure('Name','Player 2 Parameter Learning');
    tlo2 = tiledlayout(3,1,'TileSpacing','none','Padding','none');

    plot_params_subplot(tlo2, idx_V2,  theta_evolution_player2, var_evolution_player2, true_params_p2, colors, lw, fs, 'V_2', [-2,6],   0:4:11, []);
    plot_params_subplot(tlo2, idx_Q2,  theta_evolution_player2, var_evolution_player2, true_params_p2, colors, lw, fs, 'Q_2', [-3,7],  -2:3:4, []);
    plot_params_subplot(tlo2, idx_R22, theta_evolution_player2, var_evolution_player2, true_params_p2, colors, lw, fs, 'R_2', [0.2,2.8], 0.5:1:2.5, 'Update index');

    % (If you want symmetric exporting, uncomment the next two lines.)
    % set(gcf, 'PaperPositionMode', 'auto');
    % print(gcf, '-dsvg', 'params_p2.svg');

end

%% ---------------- Helper Function ----------------
function plot_params_subplot(tlo, idxs, theta_evol, var_evol, true_params, colors, lw, fs, title_str, y_lim, y_ticks, xlabel_str)
% plot_params_subplot  Plot posterior mean ±2σ and ground truth for a set of indices.
%
% Inputs:
%   idxs        : vector of parameter indices to plot
%   theta_evol  : (N x d) posterior means
%   var_evol    : (N x d) posterior variances (diagonal entries)
%   true_params : (d x 1) ground truth parameter vector
%   colors      : colormap array (at least length(idxs) rows)
%   title_str   : string used in subplot annotation (e.g. 'V_1')
%   y_lim       : [ymin ymax] or [] to leave automatic
%   y_ticks     : vector of y-ticks or [] for default
%   xlabel_str  : x-label string or [] for none

    if nargin < 10 || isempty(y_lim),      y_lim = []; end
    if nargin < 11,                       y_ticks = []; end
    if nargin < 12,                       xlabel_str = ''; end

    ax = nexttile(tlo); hold(ax,'on'); grid(ax,'on');

    N = size(theta_evol,1);
    tt = 1:N;

    for k = 1:length(idxs)
        param_idx = idxs(k);
        mu    = theta_evol(:,param_idx);
        sigma = sqrt(var_evol(:,param_idx));

        % Uncertainty band (±2σ)
        fill([tt, fliplr(tt)], ...
             [mu'+2*sigma', fliplr(mu'-2*sigma')], ...
             colors(k,:), 'FaceAlpha',0.15, 'EdgeColor','none');

        % Ground truth (dashed)
        plot(tt, true_params(param_idx)*ones(1,N), '--', ...
             'Color', colors(k,:), 'LineWidth', lw);

        % Posterior mean
        plot(tt, mu, 'Color', colors(k,:), 'LineWidth', lw);

        if k < length(idxs)
            set(ax,'XTickLabels',[]);
        end
        xlim([0 N])
    end

    if ~isempty(xlabel_str)
        xlabel(xlabel_str,'Interpreter','latex','FontSize',fs);
    end
    if ~isempty(y_ticks)
        set(ax,'YTick',y_ticks);
    end
    if ~isempty(y_lim)
        ylim(ax, y_lim);
    end

    set(ax,'FontSize',fs,'TickLabelInterpreter','latex');
    text(ax,0.98,0.97,['$\hat{W}_{' title_str '}$'], ...
         'Units','normalized','Interpreter','latex', ...
         'HorizontalAlignment','right','VerticalAlignment','top', ...
         'FontSize',fs);
    box(ax,'on');
end
