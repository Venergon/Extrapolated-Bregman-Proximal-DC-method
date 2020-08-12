function [O] = objective_1D_L2(A, x, b)
    O = (1/2)*(norm(A*x - b, 2)^2);
end
