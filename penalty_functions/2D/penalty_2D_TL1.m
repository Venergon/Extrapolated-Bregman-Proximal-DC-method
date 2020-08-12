function [P] = penalty_2D_TL1(X, lambda, a)
    P = sum(arrayfun(@(x) TL1_single_element(x, lambda, a), X), 'all');
end

function [P] = TL1_single_element(x, lambda, a)
    P = lambda*(a+1)*abs(x)/(a + abs(x));
end
