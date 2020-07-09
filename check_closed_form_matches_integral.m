% A small bit of testing I did, just to check whether the closed form for
% the MCP and SCAD functions that I made matches the integral form of those
% functions

% Probably will delete once I've made sure it's ok to also delete the
% integral form of the functions

theta = 2;
lambda = 1;

n = 100;

t = linspace(-2, 2, n);

x_int = zeros(n, 1);
x_closed = zeros(n, 1);
x_paper = zeros(n, 1);

penalty_int = @(x) (penalty_MCP(x, lambda, theta));
penalty_closed = @(x) (penalty_MCP_closed_form(x, lambda, theta));
penalty_paper = @(x) (penalty_MCP_paper(x, lambda, theta));


for i=1:n
    x_int(i) = penalty_int(t(i));
    x_closed(i) = penalty_closed(t(i));
    x_paper(i) = penalty_paper(t(i));
end

diff = x_paper - x_closed;

subplot(2, 2, 1);
plot(t, x_int);
subplot(2, 2, 2);
plot(t, x_closed);
subplot(2, 2, 3);
plot(t, x_paper);
subplot(2, 2, 4);
plot(t, diff);

function [P] = penalty_SCAD(x, lambda, theta)
    P = 0;
    
    f = @(z) (min(1, max(0, (theta*lambda-z)/((theta-1)*lambda))));
    for i=1:length(x)
        P = P + integral(f, 0, abs(x(i)));
    end
    
    P = lambda*P;
end

function [P] = penalty_SCAD_closed_form(x, lambda, theta)
    P = 0;
    
    % These were calculated by integrating the function 
    % lambda*min{1, max{theta*lambda-z, 0}/((theta-1)*lambda)}dx
    % from 0 to |x|
    % theta must be > 2
    
    % Leave the multiplying by lambda to the very end
    for i=1:length(x)
        x_curr = abs(x(i));
        if x_curr < lambda
            P = P + x_curr;
        else
            P = P + lambda;

            if x_curr < theta*lambda
                P = P + (theta*lambda*(x_curr - lambda))/((theta-1)*lambda) - (x_curr^2 - lambda^2)/(2*(theta-1)*lambda);
            else
                P = P + (theta*lambda^2*(theta - 1))/((theta-1)*lambda) - (theta^2*lambda^2 - lambda^2)/(2*(theta-1)*lambda);
            end
        end
    end
    
    P = lambda*P;
end

function [P] = penalty_MCP(x, lambda, theta)
    P = 0;
    
    f = @(z) (max(0, 1 - z/(theta * lambda)));
    for i=1:length(x)
        P = P + integral(f, 0, abs(x(i)));
    end
    
    P = lambda*P;
end

function [P] = penalty_MCP_closed_form(x, lambda, theta)
    P = 0;
    
    % These were calculated by integrating the function 
    % lambda*max{1-x/(lambda*theta), 1}dx
    % from 0 to |x|
    
    % Save the multiplying by lambda to the end
    for i=1:length(x)
        x_curr = abs(x(i));
        
        if x_curr < theta*lambda
            P = P + x_curr - (x_curr)^2/(2*lambda*theta);
        else
            P = P + lambda*theta/2;
        end
    end
    
    P = lambda*P;
end

function [P] = penalty_MCP_paper(x, lambda, theta)
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

function [P] = penalty_SCAD_paper(x, lambda, theta)
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