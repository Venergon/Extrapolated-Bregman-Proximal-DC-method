function [P] = penalty_2D_abs_frobenius(X, lambda, weighting)
    P = lambda*(sum(abs(X), 'all') - weighting*norm(X, 'fro'));
end