% Test ExtendedProximalDCMethod using a randomly generated matrix of  size
% nXm, with some gaussian noise

% Compare to cvx (http://cvxr.com/cvx/), solving the L1 penalty problem in
% both cases
USE_2_NORM = true;
rtol = 1e-6;
lambda = 100;
n = 1000;
m = 2000;
density = 0.01;
noise_mu = 0;
noise_sigma = 0.1;
threshold_iterations = 10;
theta_SCAD = 5;

rng(0);

A = rand(n, m);
%lambda = L;

x_hat = sprand(m, 1, density);
b_hat = A*x_hat;

noise = normrnd(noise_mu, noise_sigma, n, 1);
b = b_hat + noise;

[f, df, L] = get_objective_function('1D-L2', A, b);

x0 = A \ b;

dg = @(x) lambda*2*x;


gamma = 0.001;
obj_fn = @(x) (objective_1D_L2(A, x, b) + penalty_1D_cauchy(x, lambda, gamma));

stop_fn = @(x_prev, x_curr, iteration) (stop_fn_base(obj_fn, rtol, x_hat, x_prev, x_curr, iteration));

obj_fn(x0)
obj_fn(x_hat)

disp('Starting Extended Bregman Proximal DC Method');
tic
thresh = get_argmin_function(lambda, 'cauchy', 'L2', threshold_iterations, 0, 0, gamma_cauchy);
x_bregman = ExtendedProximalDCMethod(f, df, L, x0, dg, thresh, stop_fn);
time_bregman = toc
disp('Finished Extended Bregman Proximal DC Method');
obj_fn(x0)
obj_fn(x_bregman)
obj_fn(x_hat)

threshold = 0.001;
density = nnz(truncate(x_bregman, threshold))
closeness = 1/2*norm(A*x_bregman - b, 2)^2

disp('Starting cvx');
tic
cvx_begin
    variable x_cvx(m, 1)
    minimize(1/2*sum_square(A*x_cvx-b)+lambda*norm(x_cvx, 1))
cvx_end
time_cvx = toc
disp('Finished cvx');

obj_x0 = obj_fn(x0);
obj_x_bregman = obj_fn(x_bregman);
obj_x_cvx = obj_fn(x_cvx);
obj_x_hat = obj_fn(x_hat);

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
