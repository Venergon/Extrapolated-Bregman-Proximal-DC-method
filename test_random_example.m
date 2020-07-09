% Test ExtendedProximalDCMethod using a randomly generated matrix of size
% nXm, with some gaussian noise
USE_2_NORM = true;
rtol = 1e-5;
lambda = 10;
n = 1000;
m = 2000;
density = 0.01;
noise_mu = 0;
noise_sigma = 0.1;
threshold_iterations = 100;
theta_MCP = 5;
theta_SCAD = 5;
a = 1;

A = rand(n, m);

x_hat = sprand(m, 1, density);
b_hat = A*x_hat;

noise = normrnd(noise_mu, noise_sigma, n, 1);
b = b_hat + noise;

x0 = A \ b;
% g = ||x||_2, so dg = 

dg_L2 = @(x) lambda*dg_2_norm(x);
dg_0 = @(x) (0);

% DC decompositions of MCP, SCAD and Transformed L1 come from 
% https://link.springer.com/article/10.1007/s10589-017-9954-1
% All three use the L1 norm as the positive convex part
dg_MCP = @(x) (lambda.*sign(x).*min(1, abs(x)/(theta_MCP*lambda)));
dg_SCAD = @(x) (sign(x).*(min(theta_SCAD*lambda, abs(x)) - lambda)/(theta_SCAD - 1));
dg_TL1 = @(x) (sign(x).*((a+1)/(a)) - sign(x).*(a^2 + a)./((a + abs(x)).^2));

obj_fn_L1_L2 = @(x) (norm(A*x-b)^2 + lambda *(norm(x, 1) - norm(x, 2)));
obj_fn_L1 = @(x) (norm(A*x-b)^2 + lambda * norm(x, 1));
obj_fn_MCP = @(x) (norm(A*x-b)^2 + penalty_MCP_closed_form(x, lambda, theta_MCP));
obj_fn_SCAD = @(x) (norm(A*x-b)^2 + penalty_SCAD_closed_form(x, lambda, theta_SCAD));
obj_fn_TL1 = @(x) (norm(A*x-b)^2 + penalty_TL1(x, a));

stop_fn = @(obj_fn)  (@(x_prev, x_curr)((obj_fn(x_curr) <= obj_fn(x_prev)) && (obj_fn(x_prev) - obj_fn(x_curr) < rtol*obj_fn(x_hat))));

stop_fn_L1_L2 = stop_fn(obj_fn_L1_L2);
stop_fn_L1 = stop_fn(obj_fn_L1);
stop_fn_MCP = stop_fn(obj_fn_MCP);
stop_fn_SCAD = stop_fn(obj_fn_SCAD);
stop_fn_TL1 = stop_fn(obj_fn_TL1);


x_L1_L2 = ExtendedProximalDCMethod(A, b, x0, dg_L2, lambda, threshold_iterations, stop_fn_L1_L2);
b_L1_L2 = A*x_L1_L2;

x_least_squares = A \ b;
b_least_squares = A*x_least_squares;

obj_hat = obj_fn_L1_L2(x_hat);
obj_L1_L2 = obj_fn_L1_L2(x_L1_L2);
obj_least_squares = obj_fn_L1_L2(x_least_squares);

b_diff_L1_L2 = norm(b_L1_L2 - b, 2)/norm(b, 2);
b_diff_hat = norm(b_hat - b, 2)/norm(b, 2);
b_diff_least_squares = norm(b_least_squares - b, 2)/norm(b, 2);

x_diff_L1_L2 = norm(x_L1_L2 - x_hat, 2)/norm(x_hat, 2);
x_diff_least_squares = norm(x_least_squares - x_hat, 2)/norm(x_hat, 2);

x_L1 = ExtendedProximalDCMethod(A, b, x0, dg_0, lambda, threshold_iterations, stop_fn_L1);
x_MCP = ExtendedProximalDCMethod(A, b, x0, dg_MCP, lambda, threshold_iterations, stop_fn_MCP);
x_SCAD = ExtendedProximalDCMethod(A, b, x0, dg_SCAD, lambda, threshold_iterations, stop_fn_SCAD);
x_TL1 = ExtendedProximalDCMethod(A, b, x0, dg_TL1, (a+1)/a, threshold_iterations, stop_fn_TL1);


% Truncate all elements below this threshold
threshold = 0.1;

% Plot the values of each version of x to see how close they are
indices = 1:m;
plot(indices, truncate(x_hat, threshold), 'x', 'DisplayName', 'Original x');
hold on;
plot(indices, truncate(x_L1_L2, threshold), 'x', 'DisplayName', 'L1 - L2');
plot(indices, truncate(x_L1, threshold), 'x', 'DisplayName', 'L1');
plot(indices, truncate(x_MCP, threshold), 'x', 'DisplayName', 'MCP');
plot(indices, truncate(x_SCAD, threshold), 'x', 'DisplayName', 'SCAD');
plot(indices, truncate(x_TL1, threshold), 'x', 'DisplayName', 'TL1');


legend('Location', 'NorthWest');

hold off;

function [dg] = dg_2_norm(x) 
    if x == 0
        dg = 0;
    else
        dg = x ./ norm(x, 2);
    end
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

function [P] = penalty_TL1(x, a)
    P = 0;
    
    for i=1:length(x)
        P = P + (a+1)*abs(x(i))/(a + abs(x(i)));
    end
end