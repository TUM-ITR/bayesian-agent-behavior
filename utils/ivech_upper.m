function P = ivech_upper(v, n)
% ivech_upper  Inverse vech operator (upper-triangular, row-wise).
%
%   P = ivech_upper(v, n)
%
% Maps a vectorized upper-triangular representation of a symmetric matrix
% back to its full (n x n) symmetric form.
%
% The vector v is assumed to follow a *row-wise upper-triangular ordering*:
%
%   v = [ P(1,1),
%         P(1,2), P(1,3), ..., P(1,n),
%         P(2,2), P(2,3), ..., P(2,n),
%         ...
%         P(n,n) ]
%
% INPUTS:
%   v : (n*(n+1)/2 x 1) vector
%       Upper-triangular entries of a symmetric matrix, stacked row-wise
%
%   n : scalar
%       Dimension of the symmetric matrix
%
% OUTPUT:
%   P : (n x n) symmetric matrix
%
% NOTES:
%   - Off-diagonal entries are mirrored to enforce symmetry.
%   - This is the inverse of the vech operator used throughout the codebase:
%         vech(P) = P(triu(true(n)))
%

    P = zeros(n);
    idx = 1;

    for i = 1:n
        for j = i:n
            P(i,j) = v(idx);
            P(j,i) = v(idx);
            idx = idx + 1;
        end
    end
end
