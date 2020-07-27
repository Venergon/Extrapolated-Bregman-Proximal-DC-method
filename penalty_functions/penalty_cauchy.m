function [P] = penalty_cauchy(x, lambda, gamma)
    P = 0;
    
    for i=1:length(x)
        P = P - lambda*log(gamma/(x(i)^2 + gamma));
    end
end
