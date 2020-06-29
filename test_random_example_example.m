% Test ExtendedProximalDCMethod using a randomly generated matrix of size
% nXm, with some gaussian noise
USE_2_NORM = true;
rtol = 1e-3;
lambda = 10;
n = 100;
m = 101;
density = 0.5;
noise_mu = 0;
noise_sigma = 0.2;

A = rand(n, m);

x_hat = sprand(m, 1, density);
b_hat = A*x_hat;

noise = normrnd(noise_mu, noise_sigma, n, 1);
b = b_hat + noise;

x0 = A \ b;
% g = ||x||_2, so dg = 


if USE_2_NORM
    dg = @(x) dg_2_norm(x);
    obj_fn = @(x) (norm(A*x-b)^2 + lambda *(norm(x, 1) - norm(x, 2)));
else
    dg = @(x) (0);
    obj_fn = @(x) (norm(A*x-b)^2 + lambda * norm(x, 1));
end
    

stop_fn = @(x_prev, x_curr)(obj_fn(x_curr) >= obj_fn(x_prev) && obj_fn(x_curr) - obj_fn(x_prev) < rtol*obj_fn(x_hat));



x_denoised = ExtendedProximalDCMethod(A, b, x0, dg, lambda, stop_fn);
b_denoised = A*x_denoised;

x_least_squares = A \ b;
b_least_squares = A*x_least_squares;

obj_hat = obj_fn(x_hat);
obj_denoised = obj_fn(x_denoised);
obj_least_squares = obj_fn(x_least_squares);

b_diff_denoised = norm(b_denoised - b, 2)/norm(b, 2);
b_diff_hat = norm(b_hat - b, 2)/norm(b, 2);
b_diff_least_squares = norm(b_least_squares - b, 2)/norm(b, 2);

x_diff_denoised = norm(x_denoised - x_hat, 2)/norm(x_hat, 2);
x_diff_least_squares = norm(x_least_squares - x_hat, 2)/norm(x_hat, 2);

function [dg] = dg_2_norm(x) 
    if x == 0
        dg = 0;
    else
        dg = x ./ norm(x, 2);
    end
end