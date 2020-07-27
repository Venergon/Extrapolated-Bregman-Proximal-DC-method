function [P] = penalty_SCAD(x, lambda, theta)
    % Version of the SCAD penalty from https://myweb.uiowa.edu/pbreheny/7600/s16/notes/2-29.pdf
    P = 0;

    for i=1:length(x)
        x_curr = abs(x(i));
        if (x_curr <= lambda)
            P = P + lambda*x_curr;
        elseif (x_curr <= lambda*theta)
            P = P + (2*lambda*theta*x_curr - x_curr^2 - lambda^2)/(2*(theta-1));
        else
            P = P + (lambda^2*(theta+1))/2;
        end
    end
end