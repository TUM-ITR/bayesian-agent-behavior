function recon = compute_reconstruction(theta_p1, theta_p2, true_p1, true_p2, ...
                                        R11, R22, B1, B2, A, n_x, n_u1, n_u2)
% compute_reconstruction  Reconstruct value/cost matrices and feedback gains from estimates.
%
%   recon = compute_reconstruction(theta_p1, theta_p2, true_p1, true_p2, ...
%                                  R11, R22, B1, B2, A, n_x, n_u1, n_u2)
%
% Given the history of estimated parameter vectors for both players, this
% function extracts the final estimates (last time step) and reconstructs:
%   - P1_hat, P2_hat from vech(P_i)
%   - R11_hat, R22_hat assuming diagonal R and known first diagonal entry
%   - Feedback gains K1_hat, K2_hat via K_i = R_ii^{-1} B_i' P_i
%
% Assumed parameter ordering (consistent with the main script):
%   theta_i = [ vech(P_i) ; diag(Q_i) ; diag(R_ii_free) ]
% where R_ii_free excludes the known entry R_ii(1,1), i.e. only the remaining
% diagonal inputs are estimated.
%
% INPUTS:
%   theta_p1, theta_p2 : (N x d_i) matrices
%       Time evolution of parameter estimates for player 1 and 2.
%       The final estimate is taken from the last row (end,:).
%
%   true_p1, true_p2   : (d_i x 1) vectors (currently unused)
%       Included for convenience / potential diagnostics (e.g. error metrics).
%
%   R11, R22           : (n_u1 x n_u1), (n_u2 x n_u2) matrices
%       True (or known) input cost matrices; only the scalar Rii(1,1) is used
%       here as a known fixed entry.
%
%   B1, B2             : (n_x x n_u1), (n_x x n_u2) matrices
%       Input matrices.
%
%   A                  : (n_x x n_x) matrix (currently unused)
%       Included for potential closed-loop checks / extensions.
%
%   n_x, n_u1, n_u2    : scalars
%       State and input dimensions.
%
% OUTPUT:
%   recon : struct with fields
%       .P1_hat, .P2_hat   : (n_x x n_x) reconstructed symmetric value matrices
%       .R11_hat, .R22_hat : (n_ui x n_ui) reconstructed diagonal input cost matrices
%       .K1_hat, .K2_hat   : (n_ui x n_x) reconstructed feedback gains
%       .indices           : struct containing parameter index ranges (V,Q,R) used
%
% NOTES:
%   - diag(Q_i) indices are computed but not used in this function; they are
%     returned in recon.indices for consistency with plotting/analysis.
%   - This function assumes R is diagonal and only estimates entries 2..n_u.
%

    %#ok<NASGU> % true_p1, true_p2, A are currently unused but kept for API consistency.

    % --- Parameter block sizes / indices ---
    nV = n_x * (n_x + 1) / 2;  % length(vech(P))

    % Player 1 indices
    idx_V_p1  = 1:nV;
    idx_Q_p1  = nV + (1:n_x);
    idx_R11   = nV + n_x + (1:(n_u1-1));  % estimates diag entries excluding R11(1,1)

    % Player 2 indices
    idx_V_p2  = 1:nV;
    idx_Q_p2  = nV + (1:n_x);
    idx_R22   = nV + n_x + (1:(n_u2-1));  % estimates diag entries excluding R22(1,1)

    % --- Final (last) parameter estimates ---
    theta1_hat = theta_p1(end,:)';
    theta2_hat = theta_p2(end,:)';

    % --- Reconstruct symmetric P matrices from vech(P) ---
    recon.P1_hat = ivech_upper(theta1_hat(idx_V_p1), n_x);
    recon.P2_hat = ivech_upper(theta2_hat(idx_V_p2), n_x);

    % --- Reconstruct diagonal R matrices (Rii(1,1) known, remaining estimated) ---
    R11_tail = theta1_hat(idx_R11);
    R22_tail = theta2_hat(idx_R22);

    recon.R11_hat = diag([R11(1,1); R11_tail(:)]);
    recon.R22_hat = diag([R22(1,1); R22_tail(:)]);

    % --- Reconstruct feedback gains ---
    recon.K1_hat = (recon.R11_hat \ (B1' * recon.P1_hat));
    recon.K2_hat = (recon.R22_hat \ (B2' * recon.P2_hat));

    % --- Store indices for later use (plots, reconstruction, prediction) ---
    recon.indices.p1.V = idx_V_p1;
    recon.indices.p1.Q = idx_Q_p1;
    recon.indices.p1.R = idx_R11;

    recon.indices.p2.V = idx_V_p2;
    recon.indices.p2.Q = idx_Q_p2;
    recon.indices.p2.R = idx_R22;
end
