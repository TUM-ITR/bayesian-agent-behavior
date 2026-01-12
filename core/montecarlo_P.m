function [V_mc_mean_p1, V_mc_mean_p2, V_mc_cov_p1, V_mc_cov_p2, valid_count] = ...
    montecarlo_P(Q1_mu, Q1_sig, Q2_mu, Q2_sig, R11_mu, R11_sig, R22_mu, R22_sig, ...
                 eps_floor, A, B1, B2, vech, Nmc)
% montecarlo_P  Monte Carlo moments for vech(P_i) in a 2-player LQ game.
%
%   [V_mc_mean_p1, V_mc_mean_p2, V_mc_cov_p1, V_mc_cov_p2, valid_count] = ...
%       montecarlo_P(Q1_mu, Q1_sig, Q2_mu, Q2_sig, R11_mu, R11_sig, R22_mu, R22_sig, ...
%                    eps_floor, A, B1, B2, vech, Nmc)
%
% Estimates the prior mean and covariance of the value-function parameters
% vech(P1) and vech(P2) by Monte Carlo sampling of diagonal cost matrices.
%
% In each Monte Carlo iteration, diagonal matrices Q1, Q2, R11, R22 are sampled
% (truncated below by eps_floor to ensure positivity), and the coupled algebraic
% Riccati system for the LQ differential game is solved. Samples for which the
% solver fails are discarded.
%
% INPUTS:
%   Q1_mu, Q2_mu    : (n_x x 1) vectors
%                    Means of the diagonal entries of Q1, Q2
%   Q1_sig, Q2_sig  : (n_x x 1) vectors
%                    Std. devs of the diagonal entries of Q1, Q2
%
%   R11_mu, R22_mu  : (n_u x 1) vectors
%                    Means of the diagonal entries of R11, R22
%   R11_sig, R22_sig: (n_u x 1) vectors
%                    Std. devs of the diagonal entries of R11, R22
%
%   eps_floor       : scalar
%                    Lower truncation floor for sampling diagonal entries
%
%   A               : (n_x x n_x) system matrix
%   B1, B2          : (n_x x n_u) input matrices for player 1 and 2
%
%   vech            : function handle
%                    vech(P) must return a column vector of the upper-triangular
%                    entries of symmetric matrix P (length n_x*(n_x+1)/2)
%
%   Nmc             : scalar integer
%                    Number of Monte Carlo draws (attempted)
%
% OUTPUTS:
%   V_mc_mean_p1    : (nV x 1) vector
%                    Sample mean of vech(P1)
%   V_mc_mean_p2    : (nV x 1) vector
%                    Sample mean of vech(P2)
%   V_mc_cov_p1     : (nV x nV) matrix
%                    Sample covariance of vech(P1)
%   V_mc_cov_p2     : (nV x nV) matrix
%                    Sample covariance of vech(P2)
%   valid_count     : scalar integer
%                    Number of successful solver calls (accepted samples)
%
% NOTES:
%   - Cross terms are set to zero: R12 = R21 = 0.
%   - If too few samples are valid, covariance estimates may be poor or singular.
%   - This function intentionally uses a try/catch to skip failed solves without
%     interrupting the script.
%

    % Monte Carlo estimation of expected value function parameters
    V_mc_p1 = [];
    V_mc_p2 = [];
    valid_count = 0;

    for m = 1:Nmc %#ok<NASGU>
        Q1_s  = sample_diag_from_mu_sigma(Q1_mu,  Q1_sig,  eps_floor);
        Q2_s  = sample_diag_from_mu_sigma(Q2_mu,  Q2_sig,  eps_floor);
        R11_s = sample_diag_from_mu_sigma(R11_mu, R11_sig, eps_floor);
        R22_s = sample_diag_from_mu_sigma(R22_mu, R22_sig, eps_floor);

        % No cross terms
        R12_s = zeros(size(R11_s));
        R21_s = zeros(size(R22_s));

        try
            [K2_s, K1_s, P2_s, P1_s] = ...
                basicLQ_game_gajicli(Q2_s, Q1_s, R22_s, R11_s, R21_s, R12_s, A, B2, B1); %#ok<ASGLU>

            valid_count = valid_count + 1;
            V_mc_p1(valid_count,:) = vech(P1_s).';
            V_mc_p2(valid_count,:) = vech(P2_s).';
        catch
            % Solver failed for this draw: discard sample
            continue;
        end
    end

    % Compute sample moments using accepted draws only
    V_mc_mean_p1 = mean(V_mc_p1, 1).';
    V_mc_mean_p2 = mean(V_mc_p2, 1).';
    V_mc_cov_p1  = cov(V_mc_p1);
    V_mc_cov_p2  = cov(V_mc_p2);
end
