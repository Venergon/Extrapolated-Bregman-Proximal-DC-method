function [P] = penalty_2D_abs(X, lambda)
    P = lambda*sum(abs(X), 'all');
end