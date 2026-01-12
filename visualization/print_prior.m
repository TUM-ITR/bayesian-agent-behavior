function print_prior(name, mu, sig, sample_vec)
% print_prior  Print formatted prior statistics and sampled values.
%
%   print_prior(name, mu, sig, sample_vec)
%
% Prints a compact, human-readable summary of a diagonal Gaussian prior and
% a corresponding sampled realization, in the form:
%
%   <name>:  μ = [ ... ],  σ = [ ... ],  sample = [ ... ]
%
% INPUTS:
%   name       : string or char array
%                Label for the parameter block (e.g., 'Q1', 'R11')
%
%   mu         : (n x 1) or (1 x n) vector
%                Mean values of the diagonal entries
%
%   sig        : (n x 1) or (1 x n) vector
%                Standard deviations of the diagonal entries
%
%   sample_vec : (n x 1) or (1 x n) vector
%                Sampled diagonal entries corresponding to the prior
%
% OUTPUT:
%   None (prints to the MATLAB command window).
%
% NOTES:
%   - This function assumes all inputs correspond to diagonal matrix entries.
%   - Used primarily by display_summary.m for console reporting.
%

    fprintf('%-4s:  μ = [', name);
    fprintf(' %.3g', mu);
    fprintf(' ],  σ = [');
    fprintf(' %.3g', sig);
    fprintf(' ],  sample = [');
    fprintf(' %.3g', sample_vec);
    fprintf(' ]\n');
end
