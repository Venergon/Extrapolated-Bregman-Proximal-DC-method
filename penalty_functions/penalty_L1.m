function [P] = penalty_L1(x, lambda)
    P = lambda*norm(x, 1);
end