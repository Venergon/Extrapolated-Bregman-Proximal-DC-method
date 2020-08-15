function [L] = objective_lipschitz_2D_frobenius(A, b)
    max_eigval = abs(eigs(A'*A, 1));

    % Calculate L as a lipschitz constant for the gradient of 1/2*|Ax - b|^2
    L = max_eigval;
end