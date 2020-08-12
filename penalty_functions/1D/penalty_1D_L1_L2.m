function [P] = penalty_1D_L1_L2(x, lambda, weighting)
    P = lambda*(norm(x, 1) - weighting*norm(x, 2));
end