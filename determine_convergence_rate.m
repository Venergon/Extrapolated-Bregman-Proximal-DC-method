rtol = 1e-6;
lambda = 10;
n = 4096;
m = 4096;
matrix_noise = 0.1;
density = 0.1;
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

rng(0);

% Generate a highly coherent matrix using the oversampled discrete cosine
% transform from page 27 of https://arxiv.org/pdf/2003.04124.pdf
P = m;
F = 10;
w = rand(1, P)';
dct = @(w, j) 1/sqrt(P) .* cos(2.*pi.*w.*j./F);
A_base = zeros(n, m);
for j = 1:n
    A_base(j, :) = dct(w, j);
end
A_noise = matrix_noise*rand(n, m);
A = A_base + A_noise;

x_hat = sprand(m, 1, density);
% Normalise x_hat to have maximum magnitude of 1
%if (norm(x_hat, Inf) > 1)
%    x_hat = x_hat ./ norm(x_hat, Inf);
%end
b_hat = A*x_hat;

noise = normrnd(noise_mu, noise_sigma, n, 1);
b = b_hat + noise;

[f, df, L] = get_objective_function('1D-L2', A, b);

x0 = A \ b;

obj_fn_L1 = @(x) (f(x) + penalty_1D_L1(x, lambda));
obj_fn_L1_L2 = @(x) (f(x) + penalty_1D_L1_L2(x, lambda));

dg_0 = @(x) (0);
dg_L2 = get_convex_derivative('L1-L2', lambda, 0, 0, 0, 0);

dg = dg_0;
obj_fn = obj_fn_L1;
penalty_function_name = 'L1';

argmin_fn = get_argmin_function(lambda, 'L1', 'L2', threshold_iterations, 0, 0, 0, 0, 0);

max_iter = 100000;

stop_fn_first = @(x_prev, x_curr, iteration)((iteration > max_iter) || stop_fn_base(obj_fn, rtol, x0, x_prev, x_curr, iteration));

tic
fprintf('Calculating once to get the optimal solution\n');
x_optimal = ExtrapolatedProximalDCMethod(f, df, L, x0, dg, argmin_fn, stop_fn_first);
toc

stop_fn_second = @(x_prev, x_curr, iteration)(stop_fn_with_obj_value(obj_fn, rtol, x0, x_optimal, x_prev, x_curr, iteration, max_iter));

tic
fprintf('Calculating objective values at each iteration for %s penalty\n', penalty_function_name);
x_approx = ExtrapolatedProximalDCMethod(f, df, L, x0, dg, argmin_fn, stop_fn_second);
t = toc

function [stop] = stop_fn_with_obj_value(obj_fn, rtol, x0, x_hat, x_prev, x_curr, iteration, max_iter)
    persistent obj_values;
    
    obj = obj_fn(x_curr);
    obj_prev = obj_fn(x_prev);
    obj_hat = obj_fn(x_hat);
    
    obj_err = abs(obj - obj_hat);
    obj_values(iteration+1) = obj_err;
    stop = stop_fn_base(obj_fn, rtol, x0, x_prev, x_curr, iteration);
    
    if iteration > max_iter
        stop = 1;
    end
    if stop && iteration > 0
        iterations = 2:iteration;
        %obj_values = obj_values(2:iteration);
        
        obj_diff = log(obj_values(2:iteration)) - log(obj_values(1:iteration-1));
        iter_diff = log(2:iteration) - log(1:iteration-1);
        obj_result = log(obj_values(2:iteration)) ./ log(1:iteration-1);%obj_diff; % ./ iter_diff;

        close all;
        figure();
        plot(iterations, obj_result);
        hold on;
        plot(iterations, obj_result(end).*ones(size(iterations)));
    end
end

