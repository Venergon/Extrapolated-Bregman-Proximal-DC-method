% Test ExtendedProximalDCMethod using a highly coherent randomly generated matrix of size
% nXm, with some gaussian noise
rtol = 1e-4;
lambda = 10;
n = 1024;
m = 1024;
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

x0 = A \ b;
% g = ||x||_2, so dg = 

dg_L2 = @(x) lambda*dg_2_norm(x);
dg_half_L2 = @(x) lambda*dg_2_norm(x)/2;
dg_double_L2 = @(x) lambda*dg_2_norm(x)*2;
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
obj_fn_L1_half_L2 = @(x) (1/2*norm(A*x-b)^2 + lambda *(norm(x, 1) - (1/2)*norm(x, 2)));
obj_fn_L1_double_L2 = @(x) (1/2*norm(A*x-b)^2 + lambda *(norm(x, 1) - 2*norm(x, 2)));

obj_fn_L1 = @(x) (1/2*norm(A*x-b)^2 + lambda * norm(x, 1));
obj_fn_MCP = @(x) (1/2*norm(A*x-b)^2 + penalty_MCP(x, lambda, theta_MCP));
obj_fn_SCAD = @(x) (1/2*norm(A*x-b)^2 + penalty_SCAD(x, lambda, theta_SCAD));
obj_fn_TL1 = @(x) (1/2*norm(A*x-b)^2 + penalty_TL1(x, lambda, a));
obj_fn_cauchy = @(x) (1/2*norm(A*x-b)^2 + penalty_cauchy(x, lambda, gamma_cauchy));
obj_fn_arctan = @(x) (1/2*norm(A*x-b)^2 + penalty_arctan(x, lambda, alpha_arctan, beta_arctan, gamma_arctan));

stop_fn = @(obj_fn)  (@(x_prev, x_curr, iteration)(stop_fn_base(obj_fn, rtol, x_hat, x_prev, x_curr, iteration)));

stop_fn_L1_L2 = stop_fn(obj_fn_L1_L2);
stop_fn_L1_half_L2 = stop_fn(obj_fn_L1_half_L2);
stop_fn_L1_double_L2 = stop_fn(obj_fn_L1_double_L2);

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

tic
disp('Calculating solution to arctan problem');
x_arctan = ExtendedProximalDCMethod(A, b, x0, dg_arctan, argmin_fn_arctan_lambda, stop_fn_arctan);
t_arctan = toc

tic
disp('Calculating solution to cauchy priory problem');
x_cauchy = ExtendedProximalDCMethod(A, b, x0, dg_cauchy, argmin_fn_cauchy_lambda, stop_fn_cauchy);
t_cauchy = toc

tic
disp('Calculating solution to L1-L2 problem');
x_L1_L2 = ExtendedProximalDCMethod(A, b, x0, dg_L2, argmin_fn_soft_lambda, stop_fn_L1_L2);
b_L1_L2 = A*x_L1_L2;
t_L1_L2 = toc

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

tic
disp('Calculating solution to L1 problem');
x_L1 = ExtendedProximalDCMethod(A, b, x0, dg_0, argmin_fn_soft_lambda, stop_fn_L1);
t_L1 = toc

tic
disp('Calculating solution to MCP problem');
x_MCP = ExtendedProximalDCMethod(A, b, x0, dg_MCP, argmin_fn_soft_lambda, stop_fn_MCP);
t_MCP = toc

tic
disp('Calculating solution to SCAD problem');
x_SCAD = ExtendedProximalDCMethod(A, b, x0, dg_SCAD, argmin_fn_soft_lambda, stop_fn_SCAD);
t_SCAD = toc

tic
disp('Calculating solution to TL1 problem');
x_TL1 = ExtendedProximalDCMethod(A, b, x0, dg_TL1, argmin_fn_soft_TL1, stop_fn_TL1);
t_TL1 = toc

tic
disp('Calculating solution to L1-1/2*L2 problem');
x_L1_half_L2 = ExtendedProximalDCMethod(A, b, x0, dg_half_L2, argmin_fn_soft_lambda, stop_fn_L1_half_L2);
t_L1_half_L2 = toc

tic
disp('Calculating solution to L1-2*L2 problem');
x_L1_double_L2 = ExtendedProximalDCMethod(A, b, x0, dg_double_L2, argmin_fn_soft_lambda, stop_fn_L1_double_L2);
t_L1_double_L2 = toc


% Truncate all elements below this threshold
threshold = 0.1;

% Plot the values of each version of x to see how close they are
indices = 1:m;
plot(indices, truncate(x_hat, threshold), 'o', 'DisplayName', 'Original x');
hold on;
plot(indices, x_L1_L2, 'x', 'DisplayName', 'L1 - L2');
plot(indices, x_L1, 'x', 'DisplayName', 'L1');
plot(indices, x_MCP, 'x', 'DisplayName', 'MCP');
plot(indices, x_SCAD, 'x', 'DisplayName', 'SCAD');
%plot(indices, x_TL1, 'x', 'DisplayName', 'TL1');
plot(indices, x_cauchy, 'x', 'DisplayName', 'Cauchy priory');
plot(indices, truncate(x_arctan, threshold), 'x', 'DisplayName', 'Arctan');
plot(indices, x_L1_half_L2, 'x', 'DisplayName', 'L1-1/2*L2');
plot(indices, x_L1_double_L2, 'x', 'DisplayName', 'L1-2*L2');


legend('Location', 'NorthWest');

hold off;


dense_x_hat = nnz(truncate(x_hat, threshold));
dense_L1_L2 = nnz(truncate(x_L1_L2, threshold));
dense_L1 = nnz(truncate(x_L1, threshold));
dense_MCP = nnz(truncate(x_MCP, threshold));
dense_SCAD = nnz(truncate(x_SCAD, threshold));
dense_TL1 = nnz(truncate(x_TL1, threshold));
dense_cauchy = nnz(truncate(x_cauchy, threshold));
dense_arctan = nnz(truncate(x_arctan, threshold));
dense_L1_half_L2 = nnz(truncate(x_L1_half_L2, threshold));
dense_L1_double_L2 = nnz(truncate(x_L1_double_L2, threshold));


function [dg] = dg_2_norm(x) 
    if x == 0
        dg = 0;
    else
        dg = x ./ norm(x, 2);
    end
end