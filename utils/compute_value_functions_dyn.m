function [V1_fun, V2_fun, results] = compute_value_functions_dyn(f, Q1, Q2, R1, R2, opts)
% COMPUTE_VALUE_FUNCTIONS_DYN  Approximate value functions for a 1D nonlinear 2-player game
%
%   [V1_fun, V2_fun, results] = compute_value_functions_dyn(f, Q1, Q2, R1, R2, opts)
%
% This routine computes approximate value functions V1(x), V2(x) for a
% continuous-time, infinite-horizon, discounted (rho > 0) 1D nonlinear
% two-player differential game with scalar dynamics:
%
%     x_dot = f(x) + g1(x) u1 + g2(x) u2 .
%
% Each player i has (scalar) running cost:
%     l_i(x, u1, u2) = Q_i * x^2 + R_i * u_i^2 .
%
% The value functions are approximated with a truncated Legendre basis over
% x in [-L, L]. We enforce exact origin constraints:
%     V_i(0) = 0,    (d/dx) V_i(0) = 0
% by parameterizing the weights in the nullspace of these constraints
% (no reliance on a specific grid index).
%
% The unknown coefficients are computed by a nonlinear least-squares fit of
% the coupled HJB residuals evaluated on a collocation grid.
%
% Inputs
%   f       : function handle, drift term f(x)
%   Q1,Q2   : positive scalars (state cost weights)
%   R1,R2   : positive scalars (input cost weights)
%   opts    : (optional) struct with fields:
%       .K        (15)     number of Legendre basis functions
%       .L        (3)      domain half-width (x in [-L, L])
%       .M        (201)    collocation grid points on [-L, L]
%       .rho      (1e-3)   discount rate
%       .g1       (@(x)1)  input channel for player 1
%       .g2       (@(x)0.8)input channel for player 2
%       .showPlot (true)   plot V_i, dV_i/dx and u_i^*
%
% Outputs
%   V1_fun  : function handle evaluating V1(x) on query points x
%   V2_fun  : function handle evaluating V2(x) on query points x
%   results : struct with fields:
%       .w1, .w2       basis weights for V1, V2
%       .Phi_fun       handle returning [Phi(x), dPhi_dx(x)]
%       .errInf        max absolute residuals on dense plotting grid
%       .xgrid, .V1, .V2, .V1x, .V2x, .u1, .u2 (if you want to store them)
%
% Notes
%   - Any constant factors (e.g., 1/2) in the optimal control expressions
%     depend on the normalization used in the paper / feature definitions.
%   - This function requires Optimization Toolbox for lsqnonlin.
%
% -------------------------------------------------------------------------

    if nargin < 6, opts = struct(); end
    if ~isfield(opts,'K'),        opts.K = 15;       end
    if ~isfield(opts,'L'),        opts.L = 3;        end
    if ~isfield(opts,'M'),        opts.M = 201;      end
    if ~isfield(opts,'rho'),      opts.rho = 1e-3;   end
    if ~isfield(opts,'showPlot'), opts.showPlot = true; end
    if ~isfield(opts,'g1'),       opts.g1 = @(x) 1.0 + 0*x; end
    if ~isfield(opts,'g2'),       opts.g2 = @(x) 0.8 + 0*x; end

    % Regularization strength for the decision variables (z), not the weights (w)
    lambda_reg = 1e-6;

    % --- Grid & basis on [-L, L]
    L = opts.L; K = opts.K; M = opts.M;
    x = linspace(-L, L, M).';
    s = x / L;

    [Phi, dPhi_ds] = legendre_basis_and_deriv(s, K);
    dPhi_dx = (1/L) * dPhi_ds;

    % --- Exact origin constraints (no reliance on grid index)
    % Enforce: V_i(0) = 0 and dV_i/dx(0) = 0 for both players i=1,2.
    s0 = 0; % x=0 -> s=0 exactly
    [Phi0, dPhi0_ds] = legendre_basis_and_deriv(s0, K);  % returns 1xK
    dPhi0_dx = (1/L) * dPhi0_ds;

    Aeq = [Phi0; dPhi0_dx];          % 2 x K
    N = null(Aeq, 'r');              % K x (K-r), r <= 2
    if isempty(N)
        error('Nullspace empty: reduce constraints or increase K.');
    end

    % --- Pack parameters for residual evaluation
    par.x       = x;
    par.Phi     = Phi;
    par.dPhi_dx = dPhi_dx;

    par.Q1  = Q1;  par.Q2  = Q2;
    par.R1  = R1;  par.R2  = R2;
    par.f   = f;
    par.g1  = opts.g1;
    par.g2  = opts.g2;
    par.rho = opts.rho;

    par.N = N;
    par.K = K;

    % --- Optimize over nullspace variables z = [z1; z2]
    kz = size(N,2);
    z0 = zeros(2*kz, 1);

    fun = @(z) residuals_game_dyn_null(z, par, lambda_reg);

    opts_lsq = optimoptions('lsqnonlin', ...
        'Display', 'off', ...
        'MaxIterations', 400, ...
        'FunctionTolerance', 1e-10, ...
        'StepTolerance', 1e-10, ...
        'MaxFunctionEvaluations', 1e5);

    zsol = lsqnonlin(fun, z0, [], [], opts_lsq);

    % Recover full weight vectors w1, w2 in the original basis
    z1 = zsol(1:kz);
    z2 = zsol(kz+1:end);
    w1 = N * z1;
    w2 = N * z2;

    % --- Evaluate on a dense plotting grid
    xplot = linspace(-L, L, 1001).';
    [PhiP, dPhiP_dx] = local_legendre_eval(xplot, L, K);

    V1  = PhiP * w1;
    V2  = PhiP * w2;
    V1x = dPhiP_dx * w1;
    V2x = dPhiP_dx * w2;

    fx  = f(xplot);
    g1x = opts.g1(xplot);
    g2x = opts.g2(xplot);

    % "Optimal" controls implied by the chosen normalization
    u1 = -0.5 * (g1x ./ R1) .* V1x;
    u2 = -0.5 * (g2x ./ R2) .* V2x;

    rho = opts.rho;

    % Coupled residuals (as used by the paper's formulation)
    Res1 = rho*V1 - ( Q1*xplot.^2 + V1x.*fx ...
          - (g1x.^2./(4*R1)).*V1x.^2 ...
          - (g2x.^2./(2*R2)).*(V1x.*V2x) );

    Res2 = rho*V2 - ( Q2*xplot.^2 + V2x.*fx ...
          - (g2x.^2./(4*R2)).*V2x.^2 ...
          - (g1x.^2./(2*R1)).*(V1x.*V2x) );

    errInf1 = max(abs(Res1));
    errInf2 = max(abs(Res2));

    % --- Outputs
    V1_fun = @(xq) eval_basis(xq, L, K, w1);
    V2_fun = @(xq) eval_basis(xq, L, K, w2);

    results.w1 = w1;
    results.w2 = w2;

    % Provide a unified basis evaluation handle consistent with callers:
    %   [Phi, dPhi_dx] = results.Phi_fun(xq)
    results.Phi_fun = @(xq) local_legendre_eval(xq, L, K);

    results.errInf  = [errInf1, errInf2];

    % Optional: stash for downstream debugging/plots
    results.xplot = xplot;
    results.V1 = V1;   results.V2 = V2;
    results.V1x = V1x; results.V2x = V2x;
    results.u1 = u1;   results.u2 = u2;
    results.Res1 = Res1; results.Res2 = Res2;

    % ===========================================================
    %   Plot value functions, gradients, and implied optimal controls
    % ===========================================================
    if opts.showPlot
        figure('Color','w','Name','Value functions, gradients, and optimal controls');
        tiledlayout(3,2,'TileSpacing','compact','Padding','compact');

        % --- Player 1: V1 ---
        nexttile; hold on; grid on;
        plot(xplot, V1, 'LineWidth', 1.6, 'Color', [0.1 0.4 0.8]);
        yline(0,'--k'); xline(0,'--k');
        xlabel('$x$','Interpreter','latex');
        ylabel('$V_1(x)$','Interpreter','latex');
        title(sprintf('$V_1$ (L$_\\infty$=%.1e)', errInf1),'Interpreter','latex');

        % --- Player 2: V2 ---
        nexttile; hold on; grid on;
        plot(xplot, V2, 'LineWidth', 1.6, 'Color', [0.8 0.2 0.1]);
        yline(0,'--k'); xline(0,'--k');
        xlabel('$x$','Interpreter','latex');
        ylabel('$V_2(x)$','Interpreter','latex');
        title(sprintf('$V_2$ (L$_\\infty$=%.1e)', errInf2),'Interpreter','latex');

        % --- Player 1: dV1/dx ---
        nexttile; hold on; grid on;
        plot(xplot, V1x, 'LineWidth', 1.6, 'Color', [0.1 0.5 0.8]);
        yline(0,'--k'); xline(0,'--k');
        xlabel('$x$','Interpreter','latex');
        ylabel('$\partial_x V_1$','Interpreter','latex');
        title('$V_1''(x)$ (gradient)','Interpreter','latex');

        % --- Player 2: dV2/dx ---
        nexttile; hold on; grid on;
        plot(xplot, V2x, 'LineWidth', 1.6, 'Color', [0.9 0.3 0.1]);
        yline(0,'--k'); xline(0,'--k');
        xlabel('$x$','Interpreter','latex');
        ylabel('$\partial_x V_2$','Interpreter','latex');
        title('$V_2''(x)$ (gradient)','Interpreter','latex');

        % --- Player 1: u1*(x) ---
        nexttile; hold on; grid on;
        plot(xplot, u1, 'LineWidth', 1.6, 'Color', [0.1 0.5 0.8]);
        yline(0,'--k'); xline(0,'--k');
        xlabel('$x$','Interpreter','latex');
        ylabel('$u_1^*(x)$','Interpreter','latex');
        title('$u_1^*(x)$','Interpreter','latex');

        % --- Player 2: u2*(x) ---
        nexttile; hold on; grid on;
        plot(xplot, u2, 'LineWidth', 1.6, 'Color', [0.9 0.3 0.1]);
        yline(0,'--k'); xline(0,'--k');
        xlabel('$x$','Interpreter','latex');
        ylabel('$u_2^*(x)$','Interpreter','latex');
        title('$u_2^*(x)$','Interpreter','latex');

        % % Zoom near origin (sanity check for constraints)
        % figure('Color','w','Name','Value functions near origin');
        % hold on; grid on;
        % plot(xplot, V1, 'LineWidth', 1.6, 'Color', [0.1 0.5 0.8]);
        % plot(xplot, V2, 'LineWidth', 1.6, 'Color', [0.9 0.3 0.1]);
        % xlim([-0.5, 0.5]); yline(0,'--k');
        % xlabel('$x$','Interpreter','latex');
        % ylabel('$V_i(x)$','Interpreter','latex');
        % legend({'$V_1(x)$','$V_2(x)$'},'Interpreter','latex','Location','best');
        % title('Value functions near origin','Interpreter','latex');
    end

    fprintf('Done: Q=[%.2f,%.2f], R=[%.2f,%.2f], residuals L∞=[%.2e, %.2e]\n', ...
        Q1, Q2, R1, R2, errInf1, errInf2);
end

% ---------------- helper functions --------------------------------------
% ===== residual in nullspace variables =====
function R = residuals_game_dyn_null(z, par, lambda_reg)
    N  = par.N;
    kz = size(N,2);

    z1 = z(1:kz);
    z2 = z(kz+1:end);

    w1 = N * z1;
    w2 = N * z2;

    Phi     = par.Phi;
    dPhi_dx = par.dPhi_dx;
    x       = par.x;

    V1  = Phi * w1;
    V2  = Phi * w2;
    V1x = dPhi_dx * w1;
    V2x = dPhi_dx * w2;

    fx  = par.f(x);
    g1x = par.g1(x);
    g2x = par.g2(x);

    rho = par.rho;
    Q1  = par.Q1;  Q2 = par.Q2;
    R1  = par.R1;  R2 = par.R2;

    HJB1 = rho*V1 - ( Q1*x.^2 + V1x.*fx ...
         - (g1x.^2./(4*R1)).*V1x.^2 ...
         - (g2x.^2./(2*R2)).*(V1x.*V2x) );

    HJB2 = rho*V2 - ( Q2*x.^2 + V2x.*fx ...
         - (g2x.^2./(4*R2)).*V2x.^2 ...
         - (g1x.^2./(2*R1)).*(V1x.*V2x) );

    R = [HJB1; HJB2];

    % Regularize z (not w) to stabilize the nullspace parametrization
    if lambda_reg > 0
        R = [R; sqrt(lambda_reg) * z];
    end
end

% ===== small utilities (unchanged) =====
function [Phi, dPhi_dx] = local_legendre_eval(xq, L, K)
    s = xq / L;
    [Phi, dPhi_ds] = legendre_basis_and_deriv(s, K);
    dPhi_dx = (1/L) * dPhi_ds;
end

function [P, dP_ds] = legendre_basis_and_deriv(s, K)
    M = numel(s);
    P = zeros(M, K);
    dP_ds = zeros(M, K);

    P(:,1) = 1;
    if K == 1
        return;
    end

    P(:,2) = s;
    dP_ds(:,2) = 1;

    for n = 1:K-2
        P(:,n+2) = ((2*n+1)*s.*P(:,n+1) - n*P(:,n)) / (n+1);
        dP_ds(:,n+2) = ((2*n+1)*(P(:,n+1) + s.*dP_ds(:,n+1)) - n*dP_ds(:,n)) / (n+1);
    end
end

function V = eval_basis(xq, L, K, w)
    [Phi, ~] = local_legendre_eval(xq, L, K);
    V = Phi * w;
end
