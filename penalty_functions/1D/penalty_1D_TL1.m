function [P] = penalty_1D_TL1(x, lambda, a)
    P = 0;
    
    for i=1:length(x)
        P = P + (a+1)*abs(x(i))/(a + abs(x(i)));
    end
    
    P = lambda*P;
end