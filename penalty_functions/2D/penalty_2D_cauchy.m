function [P] = penalty_2D_cauchy(X, lambda, gamma)
    P = sum(arrayfun(@(x) cauchy_single_element(x, lambda, gamma), X), 'all');
end

function [P] = cauchy_single_element(x, lambda, gamma)
    P = -lambda*log(gamma/x^2 + gamma);
end
