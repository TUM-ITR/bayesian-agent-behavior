function plot_state_trajectories(t, x_true, x_nom, x_ref_fun)
% plot_state_trajectories  Plot state trajectories against a reference.
%
%   plot_state_trajectories(t, x_true, x_nom, x_ref_fun)
%
% Plots each state component over time for:
%   - the reference trajectory x_ref(t)
%   - the nominal closed-loop trajectory (x_nom)
%   - the true closed-loop trajectory (x_true)
%
% INPUTS:
%   t        : (N x 1) vector
%              Time grid
%
%   x_true   : (N x n_x) matrix
%              Trajectory produced using the "true" feedback gains
%
%   x_nom    : (N x n_x) matrix
%              Trajectory produced using the "nominal" feedback gains
%
%   x_ref_fun: function handle
%              x_ref_fun(t) returns the reference state at time t (n_x x 1)
%
% OUTPUT:
%   None (creates a tiled figure).
%
% NOTES:
%   - In the main script, the dynamics are written in deviation coordinates
%     x_dev = x - x_ref(t). The variables x_true/x_nom are therefore interpreted
%     as deviations from the reference, and the plotted reference is shown for
%     comparison. (Keep this in mind when interpreting absolute levels.)
%

    % Number of states
    nStates = size(x_true, 2);

    figure('Name', 'State Trajectories: True vs Nominal vs Reference');
    tiledlayout(nStates, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    % Evaluate reference trajectory on the time grid
    x_ref_mat = cell2mat(arrayfun(@(ti) x_ref_fun(ti)', t, 'UniformOutput', false));

    for k = 1:nStates
        nexttile; hold on; grid on;
        plot(t, x_ref_mat(:,k), 'k--', 'LineWidth', 1.5);
        plot(t, x_nom(:,k),  'LineWidth', 1.5, 'Color', [0.3 0.3 0.8]);
        plot(t, x_true(:,k), 'LineWidth', 1.5, 'Color', [0.8 0.2 0.2]);

        ylabel(sprintf('$x_%d(t)$', k), 'Interpreter', 'latex');

        if k == nStates
            xlabel('Time [s]', 'Interpreter', 'latex');
        end

        title(sprintf('State $x_%d$', k), 'Interpreter', 'latex');
    end

    legend({'Reference', '$x_{nom}$', '$x_{true}$'}, ...
           'Interpreter', 'latex', 'Location', 'best');
end
