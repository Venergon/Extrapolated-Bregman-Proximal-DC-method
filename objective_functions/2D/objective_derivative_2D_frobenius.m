function [dO] = objective_derivative_2D_frobenius(A, x, b)
    dO = A'*(A*x - b);
end
