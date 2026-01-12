function x_ref = get_reference_vector(t, t_switch, x_ref_list)
% get_reference_vector  Piecewise-constant reference trajectory.
%
%   x_ref = get_reference_vector(t, t_switch, x_ref_list)
%
% Returns the reference state vector x_ref(t) for a piecewise-constant
% reference trajectory defined over switching times.
%
% INPUTS:
%   t           : scalar
%                 Current time
%
%   t_switch    : (1 x K) or (K x 1) vector
%                 Monotonically increasing switching times
%                 (e.g. [t0, t1, t2, ..., tK])
%
%   x_ref_list  : (n_x x K) matrix
%                 Reference vectors associated with each interval.
%                 Column k corresponds to the reference active on
%                 [t_switch(k), t_switch(k+1)).
%
% OUTPUT:
%   x_ref       : (n_x x 1) vector
%                 Reference state at time t
%
% BEHAVIOUR:
%   - If t <= t_switch(1), the first reference is returned.
%   - If t >= t_switch(end), the last reference is returned.
%   - For intermediate times, the reference is selected according to:
%         t_switch(k) <= t < t_switch(k+1)
%
% NOTES:
%   - The reference is piecewise constant (zero derivative except at switches).
%   - This function performs clamping outside the defined time range.
%

    % Clamp to first reference
    if t <= t_switch(1)
        x_ref = x_ref_list(:,1);
        return;
    % Clamp to last reference
    elseif t >= t_switch(end)
        x_ref = x_ref_list(:,end);
        return;
    end

    % Find active interval index
    idx = find(t >= t_switch(1:end-1) & t < t_switch(2:end), 1, 'last');
    if isempty(idx)
        idx = 1;
    end

    % Return corresponding reference vector
    x_ref = x_ref_list(:,idx);
end
