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

penalty_int = @(x) (penalty_MCP(x, lambda, theta));
penalty_closed = @(x) (penalty_MCP_closed_form(x, lambda, theta));

for i=1:n
    x_int(i) = penalty_int(t(i));
    x_closed(i) = penalty_closed(t(i));
end

diff = x_int - x_closed;

subplot(2, 2, 1);
plot(t, x_int);
subplot(2, 2, 2);
plot(t, x_closed);
subplot(2, 2, 3);
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
    
    % Leave the multiplying by lambda to the very end
    for i=1:length(x)
        x_curr = abs(x(i));
        if theta >= 1
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
        else % theta < 1
            P = P + min(x_curr, theta*lambda);
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