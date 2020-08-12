function [P] = penalty_1D_L1(x, lambda)
    P = lambda*norm(x, 1);
end