% Test ExtendedProximalDCMethod using a randomly generated matrix of size
% nXm, with some gaussian noise
rtol = 0.75e-3;
lambda = 10;
n = 1000;
m = 2000;
density = 0.01;
noise_mu = 0;
noise_sigma = 0.1;
threshold_iterations = 10;
theta_MCP = 5;
theta_SCAD = 5;
a = 1;
gamma_cauchy = 2;

beta_arctan = sqrt(3)/3;
gamma_arctan = pi/6;
alpha_arctan = 1;

M_arctan = (3*alpha_arctan^2*beta_arctan^(2/3))/(4*gamma_arctan);


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

dg_cauchy = @(x) lambda*2*x;
dg_arctan = @(x) lambda*M_arctan*x;

obj_fn_L1_L2 = @(x) (1/2*norm(A*x-b)^2 + lambda *(norm(x, 1) - norm(x, 2)));
obj_fn_L1 = @(x) (1/2*norm(A*x-b)^2 + lambda * norm(x, 1));
obj_fn_MCP = @(x) (1/2*norm(A*x-b)^2 + penalty_MCP_paper(x, lambda, theta_MCP));
obj_fn_SCAD = @(x) (1/2*norm(A*x-b)^2 + penalty_SCAD_paper(x, lambda, theta_SCAD));
obj_fn_TL1 = @(x) (1/2*norm(A*x-b)^2 + penalty_TL1(x, a));
obj_fn_cauchy = @(x) (1/2*norm(A*x-b)^2 + penalty_cauchy(x, lambda, gamma_cauchy));
obj_fn_arctan = @(x) (1/2*norm(A*x-b)^2 + penalty_arctan(x, lambda, alpha_arctan, beta_arctan, gamma_arctan));

stop_fn = @(obj_fn)  (@(x_prev, x_curr, iteration)(stop_fn_base(obj_fn, rtol, x_hat, x_prev, x_curr, iteration)));

stop_fn_L1_L2 = stop_fn(obj_fn_L1_L2);
stop_fn_L1 = stop_fn(obj_fn_L1);
stop_fn_MCP = stop_fn(obj_fn_MCP);
stop_fn_SCAD = stop_fn(obj_fn_SCAD);
stop_fn_TL1 = stop_fn(obj_fn_TL1);
stop_fn_cauchy = stop_fn(obj_fn_cauchy);
stop_fn_arctan = stop_fn(obj_fn_arctan);


argmin_fn_soft_lambda = get_argmin_function(lambda, 'L1', 'L2', threshold_iterations);
argmin_fn_soft_TL1 = get_argmin_function((a+1)/a, 'L1', 'L2', threshold_iterations);
argmin_fn_cauchy_lambda = get_argmin_function(lambda, 'cauchy', 'L2', threshold_iterations);
argmin_fn_arctan_lambda = get_argmin_function(lambda, 'arctan', 'L2', threshold_iterations);

disp('Calculating solution to arctan problem');
x_arctan = x0;%ExtendedProximalDCMethod(A, b, x0, dg_arctan, argmin_fn_arctan_lambda, stop_fn_arctan);

disp('Calculating solution to cauchy priory problem');
x_cauchy = ExtendedProximalDCMethod(A, b, x0, dg_cauchy, argmin_fn_cauchy_lambda, stop_fn_cauchy);

disp('Calculating solution to L1-L2 problem');
x_L1_L2 = ExtendedProximalDCMethod(A, b, x0, dg_L2, argmin_fn_soft_lambda, stop_fn_L1_L2);
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

disp('Calculating solution to L1 problem');
x_L1 = ExtendedProximalDCMethod(A, b, x0, dg_0, argmin_fn_soft_lambda, stop_fn_L1);
disp('Calculating solution to MCP problem');
%x_MCP = ExtendedProximalDCMethod(A, b, x0, dg_MCP, argmin_fn_soft_lambda, stop_fn_MCP);
disp('Calculating solution to SCAD problem');
%x_SCAD = ExtendedProximalDCMethod(A, b, x0, dg_SCAD, argmin_fn_soft_lambda, stop_fn_SCAD);
disp('Calculating solution to TL1 problem');
%x_TL1 = ExtendedProximalDCMethod(A, b, x0, dg_TL1, argmin_fn_soft_TL1, stop_fn_TL1);



% Truncate all elements below this threshold
threshold = 0.1;

% Plot the values of each version of x to see how close they are
indices = 1:m;
%plot(indices, truncate(x_hat, threshold), 'x', 'DisplayName', 'Original x');
hold on;
plot(indices, truncate(x_L1_L2, threshold), 'x', 'DisplayName', 'L1 - L2');
plot(indices, truncate(x_L1, threshold), 'x', 'DisplayName', 'L1');
%plot(indices, truncate(x_MCP, threshold), 'x', 'DisplayName', 'MCP');
%plot(indices, truncate(x_SCAD, threshold), 'x', 'DisplayName', 'SCAD');
%plot(indices, truncate(x_TL1, threshold), 'x', 'DisplayName', 'TL1');
plot(indices, truncate(x_cauchy, threshold), 'x', 'DisplayName', 'Cauchy priory');
%plot(indices, truncate(x_arctan, threshold), 'x', 'DisplayName', 'Arctan');

legend('Location', 'NorthWest');

dense_x_hat = nnz(truncate(x_hat, threshold));
dense_L1_L2 = nnz(truncate(x_L1_L2, threshold));
dense_L1 = nnz(truncate(x_L1, threshold));
%dense_MCP = nnz(truncate(x_MCP, threshold));
%dense_SCAD = nnz(truncate(x_SCAD, threshold));
%dense_TL1 = nnz(truncate(x_TL1, threshold));
dense_cauchy = nnz(truncate(x_cauchy, threshold));
%dense_arctan = nnz(truncate(x_arctan, threshold));



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

function [P] = penalty_cauchy(x, lambda, gamma)
    P = 0;
    
    for i=1:length(x)
        P = P - lambda*log(gamma/(x(i)^2 + gamma));
    end
end

function [P] = penalty_arctan(x, lambda, alpha, beta, gamma)
    P = 0;
    
    for i=1:length(x)
        P = P + atan((1+alpha*abs(x(i)))) - atan(1/beta);
    end
    
    P = lambda/gamma * P;
end

function [stop] = stop_fn_base(obj_fn, rtol, x_hat, x_prev, x_curr, iteration) 
    obj_difference = obj_fn(x_prev) - obj_fn(x_curr);
    
    stop = 0;
    
     if (mod(iteration, 1000) == 0)
        fprintf('Iteration: %d\n', iteration);
        fprintf('Previous 2 obj values: %e %e\n', obj_fn(x_prev), obj_fn(x_curr));
        fprintf('Diff: %e\n', obj_fn(x_prev) - obj_fn(x_curr));
    end
    
    if (obj_difference < 0) 
        fprintf('Error: obj_difference %e is negative\n', obj_difference);
        %fprintf('Prev x %e, curr x %e diff %e\n', norm(x_prev, 2), norm(x_curr, 2), norm(x_prev - x_curr, 2));
        %throw(MException('TEST'));
    elseif (obj_difference < rtol*obj_fn(x_hat))
        stop = 1;
    end
    
end
