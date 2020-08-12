function [P] = penalty_2D_arctan(X, lambda, alpha, beta, gamma)
    P = sum(arrayfun(@(x) arctan_single_element(x, lambda, alpha, beta, gamma), X), 'all');
end

function [P] = arctan_single_element(x, lambda, alpha, beta, gamma)
    P = (lambda/gamma)*atan2((1+alpha*abs(x)), beta) - atan2(1, beta);
end
