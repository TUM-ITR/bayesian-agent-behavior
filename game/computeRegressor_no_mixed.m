function sigma_i = computeRegressor_no_mixed(features, x_dot, Bi, n_players, ui, player_index)
% computeRegressor_no_mixed  Regressor for online Bayesian estimation
%                            (diagonal input costs, no mixed terms).
%
%   sigma_i = computeRegressor_no_mixed(features, x_dot, Bi, n_players, ui, player_index)
%
% This function constructs the regressor matrix sigma_i used in the online
% Bayesian regression step for player i in a continuous-time LQ differential game.
%
% The regressor corresponds to a stacked system of equations consisting of:
%   (1) A scalar Hamilton–Jacobi–Bellman (HJB) / Bellman-error equation
%   (2) n_u first-order stationarity conditions (one per input channel)
%
% The parameter vector is assumed to be ordered as:
%   theta_i = [ vech(P_i) ; diag(Q_i) ; diag(R_ii_free) ]
% where:
%   - P_i is the value matrix of player i
%   - Q_i is the (diagonal) state cost matrix
%   - R_ii_free contains only the UNKNOWN diagonal entries of R_ii
%     (R_ii(1,1) is treated as known and excluded)
%
% Consequently:
%   - Terms involving ui(1) and R_ii(1,1) are handled in computeOutput.m
%   - Terms involving ui(2:end) appear explicitly in this regressor
%
% INPUTS:
%   features : struct with fields
%       sigmaVi     : (n_x*(n_x+1)/2 x 1)
%                     Basis for the value function (quadratic in state, all terms)
%       gradSigmaVi : (n_x x n_x*(n_x+1)/2)
%                     Jacobian of sigmaVi with respect to the state x
%       sigmaQi     : (n_x x 1)
%                     Basis for the diagonal state cost terms (x.^2)
%       sigmaRi     : ((n_players*n_u) x 1)
%                     Basis for diagonal input cost terms for all players
%
%   x_dot        : (n_x x 1)
%                  State derivative at the current time step
%
%   Bi           : (n_x x n_u)
%                  Input matrix for player i
%
%   n_players    : scalar
%                  Total number of players in the game
%                  (used to index sigmaRi)
%
%   ui           : (n_u x 1)
%                  Control input of player i
%
%   player_index : scalar in {1,...,n_players}
%                  Index of the current player
%
% OUTPUT:
%   sigma_i : ((1 + n_u) x length(theta_i)) matrix
%             Regressor matrix multiplying theta_i in the Bayesian update.
%             Row 1 corresponds to the HJB/Bellman-error equation.
%             Rows 2..(1+n_u) correspond to stationarity conditions.
%
% NOTES:
%   - Mixed input terms are excluded (diagonal R assumption).
%   - The first input channel ui(1) is associated with a known R entry and
%     therefore does not appear in the R_ii_free block of the regressor.
%   - The structure of sigma_i is consistent with computeOutput.m and the
%     chosen parameter vector ordering.
%

    n_u = size(Bi,2);
    n_x = size(Bi,1);

    delete_idx = (player_index - 1) * n_u + 1;

    sigma_i = [(features.gradSigmaVi * x_dot)',       features.sigmaQi',        features.sigmaRi([delete_idx+1:delete_idx+n_u-1])'           ;                                                     
                     Bi'*features.gradSigmaVi',          zeros(n_u,n_x),        [zeros(1,n_u-1); 2*diag(ui(2:end))]                          ];
end
