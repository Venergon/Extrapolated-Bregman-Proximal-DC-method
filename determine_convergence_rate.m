% Determine the convergence rate of a specific penalty function using a
% random matrix
rng(0);

rtol = 1e-10;
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

M_arctan = (2*alpha_arctan^2*beta_arctan)/(gamma_arctan*(1+beta_arctan^2));

A = rand(n, m);

x_hat = sprand(m, 1, density);
b_hat = A*x_hat;

noise = normrnd(noise_mu, noise_sigma, n, 1);
b = b_hat + noise;

[f, df, L] = get_objective_function('1D-L2', A, b);

x0 = A \ b;

argmin_fn_soft_lambda = get_argmin_function(lambda, 'L1', 'L2', threshold_iterations);
argmin_fn_soft_TL1 = get_argmin_function((a+1)/a, 'L1', 'L2', threshold_iterations);
argmin_fn_cauchy_lambda = get_argmin_function(lambda, 'cauchy', 'L2', threshold_iterations, 0, 0, gamma_cauchy);
argmin_fn_arctan_lambda = get_argmin_function(lambda, 'arctan', 'L2', threshold_iterations, alpha_arctan, beta_arctan, gamma_arctan);

penalty_function_name = 'arctan';

dg = get_convex_derivative(penalty_function_name, lambda, a, theta_MCP, theta_SCAD, M_arctan);
g = get_penalty_function(penalty_function_name, '1D', lambda, a, theta_MCP, theta_SCAD, alpha_arctan, beta_arctan, gamma_arctan, gamma_cauchy);
argmin_fn = get_argmin_fn_for_penalty(penalty_function_name, lambda, threshold_iterations, a, alpha_arctan, beta_arctan, gamma_arctan, gamma_cauchy);

obj_fn = @(x) (f(x) + g(x));

stop_fn = @(x_prev, x_curr, iteration)(stop_fn_with_obj_value(obj_fn, rtol, x0, x_hat, x_prev, x_curr, iteration));

tic
fprintf('Calculating objective values at each iteration for %s penalty\n', penalty_function_name);
x_approx = ExtendedProximalDCMethod(f, df, L, x0, dg, argmin_fn_arctan_lambda, stop_fn);
t_arctan = toc

function [stop] = stop_fn_with_obj_value(obj_fn, rtol, x0, x_hat, x_prev, x_curr, iteration)
    obj = obj_fn(x_curr);
    obj_hat = obj_fn(x_hat);
    fprintf('%d      %e     %e    %e\n', iteration, obj, obj_hat, obj - obj_hat);
    
    stop = stop_fn_base(obj_fn, rtol, x0, x_prev, x_curr, iteration);
    
end

