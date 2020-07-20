function [P] = penalty_L1_L2(x, lambda)
    P = lambda*(norm(x, 1) - norm(x, 2));
end