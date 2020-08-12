function [P] = penalty_2D_MCP(X, lambda, theta)
    P = sum(arrayfun(@(x) MCP_single_element(x, lambda, theta), X), 'all');
end

function [P] = MCP_single_element(x, lambda, theta)
    % Version of the MCP penalty from https://myweb.uiowa.edu/pbreheny/7600/s16/notes/2-29.pdf
    x = abs(x);
    
    if (x <= lambda*theta)
        P = lambda*x - x^2/(2*theta);
    else
        P = 1/2*theta*lambda^2;
    end
end
