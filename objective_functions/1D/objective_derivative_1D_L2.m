function [dO] = objective_derivative_1D_L2(A, x, b)
    dO = A'*(A*x - b);
end
