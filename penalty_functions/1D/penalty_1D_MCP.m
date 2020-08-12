function [P] = penalty_1D_MCP(x, lambda, theta)
    % Version of the MCP penalty from https://myweb.uiowa.edu/pbreheny/7600/s16/notes/2-29.pdf
    P = 0;

    for i=1:length(x)
        x_curr = abs(x(i));
        if (x_curr <= lambda*theta)
            P = P + lambda*x_curr - x_curr^2/(2*theta);
        else
            P = P + 1/2*theta*lambda^2;
        end
    end
end
