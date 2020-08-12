function [P] = penalty_2D_SCAD(X, lambda, theta)
    P = sum(arrayfun(@(x) SCAD_single_element(x, lambda, theta), X), 'all');
end

function [P] = SCAD_single_element(x, lambda, theta)
    % Version of the SCAD penalty from https://myweb.uiowa.edu/pbreheny/7600/s16/notes/2-29.pdf
    x = abs(x);
    
    if (x <= lambda)
        P = lambda*x;
    elseif (x <= lambda*theta)
        P = (2*lambda*theta*x - x^2 - lambda^2)/(2*(theta-1));
    else
        P = (lambda^2*(theta+1))/2;
    end
end
