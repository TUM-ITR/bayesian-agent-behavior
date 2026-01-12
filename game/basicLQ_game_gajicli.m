function [K2_gain, K1_gain, P2, P1, info] = basicLQ_game_gajicli(Q2, Q1, R22, R11, R21, R12, A, B2, B1, opts)
% basicLQ_game_gajicli  Solve coupled AREs for a 2-player CT LQ game (general-sum).
%
%   [K2_gain, K1_gain, P2, P1, info] = basicLQ_game_gajicli(Q2, Q1, R22, R11, R21, R12, A, B2, B1, opts)
%
% Solves the coupled continuous-time algebraic Riccati equations (CAREs) for a
% two-player linear-quadratic (LQ) differential game using an iterative scheme
% attributed to Gajic & Li / Gajic & Shen:
%   - Two auxiliary CARE solves for initialization
%   - Iterative Lyapunov updates (Algorithm 1; eqs. (11)-(12) in the reference)
%
% Dynamics:
%   x_dot = A x + B1 u1 + B2 u2
%
% Player costs (general-sum, potentially with cross input weights):
%   J1 = ∫ ( x'Q1x + u1'R11u1 + u1'R12u2 + u2'R21u1 + u2'R22u2 ) dt
%   J2 = ∫ ( x'Q2x + u1'R11u1 + u1'R12u2 + u2'R21u1 + u2'R22u2 ) dt
% (This code assumes the block structure of R is provided via R11,R12,R21,R22.)
%
% INPUTS (order matches the function signature exactly):
%   Q2, Q1 : (n_x x n_x) state cost matrices (typically symmetric PSD)
%   R22, R11, R21, R12 : control-weight blocks (sizes consistent with B1,B2)
%                        The full block matrix R = [R11 R12; R21 R22] is typically
%                        assumed SPD for well-posedness.
%   A      : (n_x x n_x) system matrix
%   B2, B1 : (n_x x n_u2), (n_x x n_u1) input matrices
%   opts   : (optional) struct with fields:
%       .MaxIters (default 500)  maximum number of iterations
%       .Tol      (default 1e-9) relative convergence tolerance on P updates
%       .Verbose  (default false) print iteration diagnostics
%
% OUTPUTS:
%   K1_gain, K2_gain : feedback gains for Nash equilibrium policies
%                      u1 = -K1_gain x,  u2 = -K2_gain x
%   P1, P2           : value matrices (Riccati-like solutions for each player)
%   info             : struct with diagnostics (iters, converged, rel_dP, Ac_eigs)
%
% DEPENDENCIES:
%   - icare, lyap (Control System Toolbox)
%
% REFERENCES:
%   Algorithm 1 in: Gajic & Li (1988); see also Gajic & Shen (1993), p. 359 ff.
%

    if nargin < 10, opts = struct(); end
    MaxIters = get_opt(opts,'MaxIters',500);
    Tol      = get_opt(opts,'Tol',1e-9);
    Verbose  = get_opt(opts,'Verbose',false);

    % ----- Define S and Z matrices as in the referenced derivation -----
    % S terms correspond to "self" input weights; Z terms capture cross-weight effects.
    S1 = B1 * (R11 \ B1');
    S2 = B2 * (R22 \ B2');
    Z1 = B1 * (R11 \ R21 * (R11 \ B1'));
    Z2 = B2 * (R22 \ R12 * (R22 \ B2'));

    % ----- Auxiliary CARE #1:  A'P1 + P1 A - P1 S1 P1 + Q1 = 0  -----
    P1 = icare(A, B1, Q1, R11);

    % ----- Auxiliary CARE #2: on (A - S1 P1) with Q2 + P1 Z1 P1 and S2 -----
    P2 = icare(A - S1*P1, B2, Q2 + P1*Z1*P1, R22);

    % Intended to provide a stabilizing initialization under the conditions of the reference.

    % ----- Iteration: Lyapunov equations (Algorithm 1 / eqs. 11-12) -----
    it = 0; conv = false;
    while it < MaxIters
        it = it + 1;
        P1_old = P1; P2_old = P2;

        Ac = A - S1*P1_old - S2*P2_old;

        % The algorithm assumes Ac is Hurwitz at each iteration.
        if any(real(eig(Ac)) >= 0)
            error('basicLQ_game_gajicli:UnstableIteration', ...
                  'Iteration produced a non-Hurwitz Ac (unstable). Check R SPD assumptions / initialization.');
        end

        % Update P1 via Lyapunov equation:
        %   Ac' P1 + P1 Ac = - ( Q1 + P1^i S1 P1^i + P2^i Z2 P2^i )
        Q1_hat = Q1 + P1_old*S1*P1_old + P2_old*Z2*P2_old;
        P1 = lyap(Ac', Q1_hat);

        % Update P2 via Lyapunov equation:
        %   Ac' P2 + P2 Ac = - ( Q2 + P1^i Z1 P1^i + P2^i S2 P2^i )
        Q2_hat = Q2 + P1_old*Z1*P1_old + P2_old*S2*P2_old;
        P2 = lyap(Ac', Q2_hat);

        % Convergence / divergence checks
        if max(abs(P1(:))) > 1e6 || max(abs(P2(:))) > 1e6
            error('basicLQ_game_gajicli:Diverged', ...
                  'Riccati iteration diverged numerically (|P| too large).');
        end

        dP = (norm(P1-P1_old,'fro') + norm(P2-P2_old,'fro')) ...
             / max(1e-12, norm(P1_old,'fro') + norm(P2_old,'fro'));

        if Verbose
            fprintf('iter %3d: rel dP = %.3e\n', it, dP);
        end

        if dP < Tol
            conv = true;
            break;
        end
    end

    % ----- Gains from first-order optimality -----
    % (In your usage, cross terms are often zero, but the solver supports them in principle.)
    K1_gain = R11 \ B1' * P1;
    K2_gain = R22 \ B2' * P2;

    info.iters     = it;
    info.converged = conv;
    info.rel_dP    = dP;
    info.Ac_eigs   = eig(A - B1*K1_gain - B2*K2_gain); % final closed-loop check
end

% -------------- small helper --------------
function v = get_opt(s, f, d)
    if isfield(s,f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
