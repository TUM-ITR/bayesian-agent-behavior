function features = computePlayerFeatures(x, uCell)
% computePlayerFeatures  Construct basis features for LQ game regression.
%
%   features = computePlayerFeatures(x, uCell)
%
% Builds the basis functions used to parameterize:
%   - the value function V_i(x) via a quadratic basis (vech-style)
%   - the state cost x'Q_i x via diagonal-only features (x.^2)
%   - the input cost terms via diagonal-only features (u.^2 for all players)
%
% This function is shared across players; player-specific selection of
% parameters/inputs is handled downstream in computeRegressor_no_mixed.m.
%
% INPUTS:
%   x     : (n_x x 1) vector
%           Current state (in this codebase, interpreted as deviation from reference)
%
%   uCell : (n_players x 1) cell array
%           uCell{p} is the control input vector u_p of player p (n_u x 1)
%
% OUTPUT (struct 'features'):
%   features.sigmaVi     : (nV x 1) vector
%       Quadratic value-function basis, matching vech(P) ordering:
%         - diagonal terms: x_i^2
%         - off-diagonal terms: 2 x_i x_j  for i < j
%       where nV = n_x*(n_x+1)/2
%
%   features.gradSigmaVi : (nV x n_x) matrix
%       Jacobian of sigmaVi with respect to x:
%         gradSigmaVi(k,:) = d(sigmaVi(k))/dx'
%       so that (gradSigmaVi * x_dot) is (nV x 1).
%
%   features.sigmaQi     : (n_x x 1) vector
%       Diagonal-only state cost basis (x.^2), used to identify diag(Q_i).
%
%   features.sigmaRi     : (n_players*n_u x 1) vector
%       Diagonal-only input cost basis for all players, stacked as:
%         [u_1.^2; u_2.^2; ...]
%
% NOTES:
%   - The factor 2 in the off-diagonal terms ensures that sigmaVi corresponds
%     to the upper-triangular (vech) parametrization of symmetric quadratic forms.
%

    n = length(x);

    % --- Value function features: vech-style quadratic basis ---
    % sigmaVi contains all unique quadratic monomials with symmetry enforced:
    %   x_i^2 for i=j, and 2*x_i*x_j for i<j.
    sigmaVi = [];
    gradSigmaVi = [];

    for i = 1:n
        for j = i:n
            if i == j
                sigmaVi = [sigmaVi; x(i)^2];
                g = zeros(1,n);
                g(i) = 2*x(i);
            else
                sigmaVi = [sigmaVi; 2*x(i)*x(j)];
                g = zeros(1,n);
                g(i) = 2*x(j);
                g(j) = 2*x(i);
            end
            gradSigmaVi = [gradSigmaVi; g];
        end
    end

    % --- State cost features: diagonal only ---
    sigmaQi = x.^2;

    % --- Input cost features: diagonal only, all players (stacked) ---
    numPlayers = length(uCell);
    sigmaRi = [];
    for p = 1:numPlayers
        u = uCell{p};
        sigmaRi = [sigmaRi; u.^2];
    end

    % --- Output struct ---
    features.sigmaVi     = sigmaVi;
    features.gradSigmaVi = gradSigmaVi;
    features.sigmaQi     = sigmaQi;
    features.sigmaRi     = sigmaRi;
end
