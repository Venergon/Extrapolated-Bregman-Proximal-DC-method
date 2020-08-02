% Test Extended Bregman Proximal DC method using an image
image_name = 'Lena512.pgm';
rng(0);

rtol = 1e-10;
mu = 0;
sigma = 50;
threshold_iterations = 10;
theta_MCP = 5;
theta_SCAD = 5;
a = 1;
gamma_cauchy = 2;
lambda = 3000;

beta_arctan = sqrt(3)/3;
gamma_arctan = pi/6;
alpha_arctan = 1;

M_arctan = (2*alpha_arctan^2*beta_arctan)/(gamma_arctan*(1+beta_arctan^2));

image = imread(strcat('images/', image_name));
image = imresize(image, 1);
image = double(image);
[height, width] = size(image);

subplot(2, 2, 1);
imshow(uint8(image));
title('Original Image');

noise = normrnd(mu, sigma, height, width);
noisy_image = image + noise;

subplot(2, 2, 2);
imshow(uint8(noisy_image));
title(sprintf('Noisy Image, PSNR = %2.2f dB', psnr(uint8(noisy_image), uint8(image))));


image_vector = reshape(noisy_image, height*width, 1);
transformed_image_vector_complex = fft(image_vector);
transformed_image_vector = split_complex(transformed_image_vector_complex);
A = speye(length(transformed_image_vector));
b = transformed_image_vector;

x_hat = transformed_image_vector;
x0 = transformed_image_vector;

dg_L2 = @(x) lambda*dg_2_norm(x);
dg_half_L2 = @(x) lambda*dg_2_norm(x)/2;
dg_double_L2 = @(x) lambda*dg_2_norm(x)*2;
dg_0 = @(x) (0);

% DC decompositions of MCP, SCAD and Transformed L1 come from 
% https://link.springer.com/article/10.1007/s10589-017-9954-1
% All three use the L1 norm as the positive convex part
dg_MCP = @(x) (lambda.*sign(x).*min(1, abs(x)/(theta_MCP*lambda)));
dg_SCAD = @(x) (sign(x).*max(min(theta_SCAD*lambda, abs(x)) - lambda, 0)/(theta_SCAD - 1));
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
disp('Calculating solution to problem');
x_approx = ExtendedProximalDCMethod(A, b, x0, dg_arctan, argmin_fn_arctan_lambda, stop_fn_arctan);
t = toc

x_approx_combined = combine_complex(x_approx);

x_untransformed = ifft(x_approx_combined);
image_approx = reshape(x_untransformed, height, width);

subplot(2, 2, 3);
imshow(uint8(image_approx));
title(sprintf('Denoised Image, PSNR = %2.2f dB', psnr(uint8(image_approx), uint8(image))));


function [dg] = dg_2_norm(x) 
    if x == 0
        dg = 0;
    else
        dg = x ./ norm(x, 2);
    end
end