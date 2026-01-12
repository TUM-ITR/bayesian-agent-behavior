function display_summary(A, B1, B2, Q1_mu, Q1_sig, Q2_mu, Q2_sig, R11_mu, R11_sig, ...
                        R22_mu, R22_sig, Q1, Q2, R11, R22, cv_Q, cv_Rd, eps_floor, ...
                        x0, T, deltaT, V_true_p1, V_nom_p1, V_true_p2, V_nom_p2, ...
                        V_mc_mean_p1, V_mc_mean_p2, valid_count, Nmc, ...
                        K1_true, K1_nom, K2_true, K2_nom)
% display_summary  Print a human-readable summary of a simulation/estimation run.
%
%   display_summary(...)
%
% Prints to the MATLAB console:
%   - System matrices (A, B1, B2)
%   - Prior parameters (mu, sigma) and sampled diagonal values for Q/R
%   - Simulation settings (x0, T, deltaT)
%   - vech(P) vectors (true vs nominal) and Monte Carlo expectation
%   - Relative deviation between nominal and true gains (K1, K2)
%
% INPUTS:
%   (Many scalar/matrix inputs passed from the main script; see main script for context.)
%   In particular:
%     Q1_mu, Q1_sig etc. are the diagonal mean/std vectors used for sampling
%     Q1, Q2, R11, R22 are the realized sampled cost matrices
%     V_true_p1 etc. are vech(P) vectors for true/nominal solutions
%     V_mc_mean_* are Monte Carlo means of vech(P)
%
% DEPENDENCIES:
%   Requires helper function print_prior(name, mu_vec, sig_vec, sampled_diag).
%
% OUTPUT:
%   None (prints to console).
%

    disp('============================================================');
    disp('           LQ DIFFERENTIAL GAME: RUN SUMMARY');
    disp('============================================================');

    disp('System matrices:');
    disp('A =');  disp(A);
    disp('B1 ='); disp(B1);
    disp('B2 ='); disp(B2);

    fprintf('\n--- Gaussian Priors (mu, sigma) and Sampled Diagonals ---\n');
    print_prior('Q1',  Q1_mu,  Q1_sig,  diag(Q1));
    print_prior('Q2',  Q2_mu,  Q2_sig,  diag(Q2));
    print_prior('R11', R11_mu, R11_sig, diag(R11));
    print_prior('R22', R22_mu, R22_sig, diag(R22));

    fprintf('\n--- Prior sampling settings ---\n');
    fprintf('cv_Q = %.2f, cv_Rd = %.2f, eps_floor = %.1e\n', cv_Q, cv_Rd, eps_floor);

    fprintf('\n--- Simulation settings ---\n');
    fprintf('x0 = [%s]\n', num2str(x0', '%.2f '));
    fprintf('T  = %.2f s,   deltaT = %.3f s\n', T, deltaT);

    fprintf('\n--- Value-function parameters (vech(P)) ---\n');
    fprintf('Player 1: true    (len=%d): [', numel(V_true_p1));
    fprintf(' %.3g', V_true_p1); fprintf(' ]\n');
    fprintf('Player 1: nominal          : [');
    fprintf(' %.3g', V_nom_p1); fprintf(' ]\n');

    fprintf('Player 2: true    (len=%d): [', numel(V_true_p2));
    fprintf(' %.3g', V_true_p2); fprintf(' ]\n');
    fprintf('Player 2: nominal          : [');
    fprintf(' %.3g', V_nom_p2); fprintf(' ]\n');

    fprintf('\n--- Monte Carlo moments for vech(P) ---\n');
    fprintf('Valid solves: %d / %d\n', valid_count, Nmc);
    fprintf('Player 1: E[vech(P1)] ≈ ['); fprintf(' %.3g', V_mc_mean_p1); fprintf(' ]\n');
    fprintf('Player 2: E[vech(P2)] ≈ ['); fprintf(' %.3g', V_mc_mean_p2); fprintf(' ]\n');

    relK1 = norm(K1_true - K1_nom, 'fro') / max(1e-12, norm(K1_nom, 'fro'));
    relK2 = norm(K2_true - K2_nom, 'fro') / max(1e-12, norm(K2_nom, 'fro'));

    fprintf('\n--- Controller deviation (true vs nominal) ---\n');
    fprintf('rel ||K1_true - K1_nom||_F = %.3f\n', relK1);
    fprintf('rel ||K2_true - K2_nom||_F = %.3f\n', relK2);

    disp('============================================================');
    disp('              Simulation completed');
    disp('============================================================');
end
