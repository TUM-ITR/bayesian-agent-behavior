function [m, S, w, K] = online_bayes_update(m, S, x, y, Sigma2)
% online_bayes_update  Online Bayesian linear regression update (Kalman form).
%
%   [m, S, w, K] = online_bayes_update(m, S, x, y, Sigma2)
%
% Performs one recursive Bayesian update for a linear-Gaussian regression model
% with possibly multi-dimensional outputs:
%
%       y = x * theta + epsilon ,   epsilon ~ N(0, Sigma2)
%
% where theta is the unknown parameter vector with Gaussian prior:
%
%       theta ~ N(m, S)
%
% The update is equivalent to a Kalman filter measurement update, applied to
% a static parameter vector theta.
%
% INPUTS:
%   m      : (d x 1) vector
%            Prior mean of the parameter vector theta
%
%   S      : (d x d) matrix
%            Prior covariance of the parameter vector theta
%
%   x      : (m_out x d) matrix
%            Regressor matrix evaluated at the current data point
%
%   y      : (m_out x 1) vector
%            Observed output (measurement)
%
%   Sigma2 : (m_out x m_out) matrix
%            Covariance of the observation noise
%
% OUTPUTS:
%   m      : (d x 1) vector
%            Posterior mean of theta after incorporating y
%
%   S      : (d x d) matrix
%            Posterior covariance of theta
%
%   w      : (d x 1) vector
%            Updated parameter estimate (equal to posterior mean m)
%
%   K      : (d x m_out) matrix
%            Kalman gain used in the update
%
% NOTES:
%   - This implementation supports multi-dimensional outputs y.
%   - No constraints (e.g. positivity) are enforced on the parameters.
%   - The update uses the Joseph-form covariance update for numerical stability.
%

    d = length(m);
    mout = length(y);

    % Predicted output
    yhat = x * m;          % (mout x 1)

    % Innovation
    nu = y - yhat;         % (mout x 1)

    % Innovation covariance
    Sy = x * S * x' + Sigma2;   % (mout x mout)

    % Kalman gain
    K = S * x' / Sy;       % (d x mout)

    % Update mean
    m = m + K * nu;

    % Update covariance (Joseph form)
    I = eye(d);
    S = (I - K*x) * S * (I - K*x)' + K*Sigma2*K';

    % Updated weights (posterior mean)
    w = m;
end
