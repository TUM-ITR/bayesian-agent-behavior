function [D, mu_vec, sig_vec] = sample_diag_gaussian(meanD, cv, eps_floor, fixed_mask)
% sample_diag_gaussian  Sample a diagonal matrix with Gaussian uncertainty.
%
%   [D, mu_vec, sig_vec] = sample_diag_gaussian(meanD, cv, eps_floor, fixed_mask)
%
% Samples a diagonal matrix D whose diagonal entries are drawn independently
% from truncated Gaussian distributions, with optional fixed (non-random)
% entries specified by a mask.
%
% INPUTS:
%   meanD      : (n x n) diagonal matrix
%                Mean (nominal) diagonal matrix
%
%   cv         : scalar
%                Coefficient of variation, defining the standard deviation as
%                sig_i = cv * |meanD(i,i)|
%
%   eps_floor  : scalar
%                Lower truncation bound to enforce positivity of sampled entries
%
%   fixed_mask : (n x 1) or (1 x n) logical vector
%                fixed_mask(i) = true  → D(i,i) is fixed to meanD(i,i)
%                fixed_mask(i) = false → D(i,i) is sampled
%
% OUTPUTS:
%   D          : (n x n) diagonal matrix
%                Sampled diagonal matrix
%
%   mu_vec     : (n x 1) vector
%                Mean values of the diagonal entries (diag(meanD))
%
%   sig_vec    : (n x 1) vector
%                Standard deviations of the diagonal entries, computed as
%                sig_vec(i) = cv * |mu_vec(i)|
%
% NOTES:
%   - Each diagonal entry is sampled independently.
%   - Rejection sampling is used to enforce D(i,i) > eps_floor.
%   - This function is used to generate both true cost matrices and
%     the corresponding mean/std vectors for Monte Carlo estimation.
%

    vals = diag(meanD);
    n = length(vals);

    D = zeros(n);
    mu_vec  = vals(:);
    sig_vec = cv * abs(vals(:));

    for i = 1:n
        if fixed_mask(i)
            % Fixed (known) diagonal entry
            D(i,i) = vals(i);
        else
            % Sampled diagonal entry (truncated Gaussian)
            mu  = mu_vec(i);
            sig = sig_vec(i);

            x = mu + sig*randn();
            while x <= eps_floor
                x = mu + sig*randn();
            end

            D(i,i) = x;
        end
    end
end
