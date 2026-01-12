function D = sample_diag_from_mu_sigma(mu_vec, sig_vec, eps_floor)
% sample_diag_from_mu_sigma  Sample a positive diagonal matrix.
%
%   D = sample_diag_from_mu_sigma(mu_vec, sig_vec, eps_floor)
%
% Samples a diagonal matrix whose diagonal entries are drawn independently
% from Gaussian distributions and truncated below by eps_floor to enforce
% strict positivity.
%
% INPUTS:
%   mu_vec   : (n x 1) or (1 x n) vector
%              Mean values of the diagonal entries
%
%   sig_vec  : (n x 1) or (1 x n) vector
%              Standard deviations of the diagonal entries
%
%   eps_floor: scalar
%              Lower bound enforced on each sampled diagonal entry
%
% OUTPUT:
%   D        : (n x n) diagonal matrix
%              Sampled diagonal matrix with strictly positive entries
%
% NOTES:
%   - Each diagonal entry is sampled independently.
%   - Rejection sampling is used to enforce positivity (x > eps_floor).
%   - This function is primarily used to sample Q and R matrices in
%     Monte Carlo estimation of prior moments.
%

    n = numel(mu_vec);
    D = zeros(n);

    for ii = 1:n
        mu = mu_vec(ii);
        sg = sig_vec(ii);

        % Rejection sampling to enforce positivity
        x = mu + sg*randn();
        while x <= eps_floor
            x = mu + sg*randn();
        end

        D(ii,ii) = x;
    end
end
