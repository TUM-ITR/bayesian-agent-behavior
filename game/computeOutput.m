function y_regr = computeOutput(r_i1, u_i)
% computeOutput  Known-output contribution for online Bayesian regression.
%
%   y_regr = computeOutput(r_i1, u_i)
%
% This function builds the part of the observation vector y that depends on the
% KNOWN control-cost weight r_i1 = R_ii(1,1) (first diagonal entry of R_ii).
% In the estimation setup used by this repo, R_ii(1,1) is treated as known and
% therefore excluded from the estimated parameter vector. The remaining diagonal
% entries R_ii(2,2),...,R_ii(n_u,n_u) are handled by the regressor (see
% computeRegressor_no_mixed.m) via terms involving u_i(2:end).
%
% INPUTS:
%   r_i1 : scalar
%       Known control-cost weight R_ii(1,1).
%   u_i  : (n_u x 1) vector
%       Current control input of player i.
%
% OUTPUT:
%   y_regr : ((n_u+1) x 1) vector
%       Observation contribution associated with the first input channel u_i(1):
%         y_regr(1) = - r_i1 * u_i(1)^2
%         y_regr(2) = - 2 * r_i1 * u_i(1)
%       The remaining entries correspond to the other input channels (2..n_u).
%       Under the diagonal-R assumption, r_i1 does not contribute to those
%       channels, so they are zero here.
%
% NOTE:
%   For n_u = 2 (the default in this script), y_regr is 3x1.
%

    n_u = length(u_i);

    y_regr = [-r_i1 * u_i(1)^2, -2 * r_i1 * u_i(1), zeros(1, n_u-1)]';
end
