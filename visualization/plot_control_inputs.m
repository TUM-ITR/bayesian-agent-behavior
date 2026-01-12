function plot_control_inputs(t, x_true, x_nom, K1_true, K1_nom, K2_true, K2_nom, x_ref_fun)
% plot_control_inputs  Plot true vs nominal control inputs for both players.
%
%   plot_control_inputs(t, x_true, x_nom, K1_true, K1_nom, K2_true, K2_nom, x_ref_fun)
%
% Reconstructs and plots the control inputs for each player using state-feedback
% laws of the form:
%   u_i(t) = -K_i * (x(t) - x_ref(t))
%
% Both the "true" controllers (computed from sampled cost matrices) and the
% "nominal" controllers (computed from prior means) are shown for comparison.
%
% INPUTS:
%   t         : (N x 1) vector
%               Time grid
%
%   x_true    : (N x n_x) matrix
%               State trajectory under true feedback gains
%
%   x_nom     : (N x n_x) matrix
%               State trajectory under nominal feedback gains
%
%   K1_true,
%   K1_nom   : (n_u1 x n_x) matrices
%               True and nominal feedback gains for player 1
%
%   K2_true,
%   K2_nom   : (n_u2 x n_x) matrices
%               True and nominal feedback gains for player 2
%
%   x_ref_fun: function handle
%               x_ref_fun(t) returns the reference state at time t (n_x x 1)
%
% OUTPUT:
%   None (creates a figure with time histories of control inputs).
%
% NOTES:
%   - Inputs are reconstructed from stored state trajectories and gains;
%     they are not directly simulated by the ODE solver.
%   - The plot layout stacks all input channels of both players vertically.
%

    % Number of input channels per player
    n_u1 = size(K1_true, 1);
    n_u2 = size(K2_true, 1);

    % Compute reference trajectory on the time grid
    x_ref_vals = arrayfun(@(ti) x_ref_fun(ti), t, 'UniformOutput', false);
    X_ref = cell2mat(x_ref_vals')';

    % State deviations from reference
    x_dev_true = x_true - X_ref;
    x_dev_nom  = x_nom  - X_ref;

    % Reconstruct control inputs
    % (note: x_dev_* are row-wise in time, hence right-multiplication by K')
    U1_true = -(x_dev_true * K1_true');
    U2_true = -(x_dev_true * K2_true');
    U1_nom  = -(x_dev_nom  * K1_nom');
    U2_nom  = -(x_dev_nom  * K2_nom');

    % Create figure
    figure('Name', 'Control Inputs: True vs Nominal (Tracking)');
    tiledlayout(n_u1 + n_u2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    % --- Player 1 inputs ---
    for k = 1:n_u1
        nexttile; hold on; grid on;
        plot(t, U1_nom(:,k),  'LineWidth', 1.5, 'Color', [0.3 0.3 0.8]);
        plot(t, U1_true(:,k), 'LineWidth', 1.5, 'Color', [0.8 0.2 0.2]);
        ylabel(sprintf('$u_{1,%d}(t)$', k), 'Interpreter', 'latex');
        title(sprintf('Player 1 Input $u_{1,%d}(t)$', k), 'Interpreter', 'latex');
    end

    % --- Player 2 inputs ---
    for k = 1:n_u2
        nexttile; hold on; grid on;
        plot(t, U2_nom(:,k),  'LineWidth', 1.5, 'Color', [0.3 0.3 0.8]);
        plot(t, U2_true(:,k), 'LineWidth', 1.5, 'Color', [0.8 0.2 0.2]);
        ylabel(sprintf('$u_{2,%d}(t)$', k), 'Interpreter', 'latex');
        title(sprintf('Player 2 Input $u_{2,%d}(t)$', k), 'Interpreter', 'latex');
        if k == n_u2
            xlabel('Time [s]', 'Interpreter', 'latex');
        end
    end

    legend({'Nominal', 'True'}, 'Interpreter', 'latex', 'Location', 'best');
end

