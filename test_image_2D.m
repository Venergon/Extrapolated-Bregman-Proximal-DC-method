% Test Extended Bregman Proximal DC method using an image
image_name = 'peppers.png';
rng(0);

rtol = 1e-10;
mu = 0;
sigma = 50;
threshold_iterations = 10;
theta_MCP = 5;
theta_SCAD = 5;
a = 1;
gamma_cauchy = 2;
lambda = 300;

beta_arctan = sqrt(3)/3;
gamma_arctan = pi/6;
alpha_arctan = 1;

M_arctan = (2*alpha_arctan^2*beta_arctan)/(gamma_arctan*(1+beta_arctan^2));

M_cauchy = 2;

image = rgb2gray(imread(strcat('images/', image_name)));
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

transformed_image = dct2(noisy_image);
%transformed_image_vector_complex = reshape(transformed_image, height*width, 1);

%image_vector = reshape(noisy_image, height*width, 1);
%transformed_image_vector_complex = fft(image_vector);
%transformed_image_vector = split_complex(transformed_image_vector_complex);
A = speye(length(transformed_image));
b = A*transformed_image;

x_hat = transformed_image;
x0 = transformed_image;

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

dg_cauchy = @(x) lambda*M_cauchy*x;
dg_arctan = @(x) lambda*M_arctan*x;

obj_fn_L1_L2 = @(x) (objective_2D_frobenius(A, x, b) + penalty_2D_abs_frobenius(x, lambda, 1));
obj_fn_L1_half_L2 = @(x) (objective_2D_frobenius(A, x, b) + penalty_2D_abs_frobenius(x, lambda, 1/2));
obj_fn_L1_double_L2 = @(x) (objective_2D_frobenius(A, x, b) + penalty_2D_abs_frobenius(x, lambda, 2));

obj_fn_L1 = @(x) (objective_2D_frobenius(A, x, b) + penalty_2D_abs(x, lambda));%penalty_L1(x, lambda));
obj_fn_MCP = @(x) (objective_2D_frobenius(A, x, b) + penalty_2D_MCP(x, lambda, theta_MCP));
obj_fn_SCAD = @(x) (objective_2D_frobenius(A, x, b) + penalty_2D_SCAD(x, lambda, theta_SCAD));
obj_fn_TL1 = @(x) (objective_2D_frobenius(A, x, b) + penalty_2D_TL1(x, lambda, a));
obj_fn_cauchy = @(x) (objective_2D_frobenius(A, x, b) + penalty_2D_cauchy(x, lambda, gamma_cauchy));
obj_fn_arctan = @(x) (objective_2D_frobenius(A, x, b) + penalty_2D_arctan(x, lambda, alpha_arctan, beta_arctan, gamma_arctan));

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
x_approx = ExtendedProximalDCMethod(A, b, x0, dg_L2, argmin_fn_soft_lambda, stop_fn_L1_L2);
t = toc

%x_approx_combined = combine_complex(x_approx);

%x_approx_reshaped = reshape(x_approx_combined, height, width);
image_approx = idct2(x_approx);
%x_untransformed = ifft(x_approx_combined);
%image_approx = reshape(x_untransformed, height, width);

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