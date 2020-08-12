function [O] = objective_2D_frobenius(A, X, B)
    O = (1/2)*(norm(A*X - B, 'fro')^2);
end
