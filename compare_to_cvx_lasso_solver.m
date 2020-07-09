% Test ExtendedProximalDCMethod using a randomly generated matrix of size
% nXm, with some gaussian noise

% Compare to cvx (http://cvxr.com/cvx/), solving the L1 penalty problem in
% both cases
USE_2_NORM = true;
rtol = 1e-4;
lambda = 100;
n = 1000;
m = 2000;
density = 0.01;
noise_mu = 0;
noise_sigma = 0.1;
threshold_iterations = 100;

A = rand(n, m);
L = norm(A'*A, 2);
%lambda = L;

x_hat = sprand(m, 1, density);
b_hat = A*x_hat;

noise = normrnd(noise_mu, noise_sigma, n, 1);
b = b_hat + noise;

x0 = A \ b;

dg = @(x) (0);

obj_fn = @(x) (norm(A*x-b, 2)^2 + lambda * norm(x, 1));


stop_fn = @(x_prev, x_curr)((obj_fn(x_curr) <= obj_fn(x_prev)) && (obj_fn(x_prev) - obj_fn(x_curr) < rtol*obj_fn(x_hat)));

obj_fn(x0)
obj_fn(x_hat)

disp('Starting Extended Bregman Proximal DC Method');
tic
x_bregman = ExtendedProximalDCMethod(A, b, x0, dg, lambda, threshold_iterations, stop_fn);
time_bregman = toc
disp('Finished Extended Bregman Proximal DC Method');
obj_fn(x0)
obj_fn(x_bregman)
obj_fn(x_hat)

disp('Starting cvx');
tic
cvx_begin
    variable x_cvx(m, 1)
    minimize(sum_square(A*x_cvx-b)+lambda*norm(x_cvx, 1))
cvx_end
time_cvx = toc
disp('Finished cvx');

obj_x0 = obj_fn(x0);
obj_x_bregman = obj_fn(x_bregman);
obj_x_cvx = obj_fn(x_cvx);
obj_x_hat = obj_fn(x_hat);
