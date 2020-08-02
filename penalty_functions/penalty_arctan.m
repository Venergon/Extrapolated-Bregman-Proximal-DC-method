function [P] = penalty_arctan(x, lambda, alpha, beta, gamma)
    P = 0;
    
    for i=1:length(x)
        P = P + atan2((1+alpha*abs(x(i))), beta) - atan2(1, beta);
    end
    
    P = lambda/gamma * P;
end
