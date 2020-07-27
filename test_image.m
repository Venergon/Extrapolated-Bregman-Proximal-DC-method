% Test Extended Bregman Proximal DC method using an image
image_name = 'Barbara.tif';
rng(0);

rtol = 1e-6;
lambda = 100;
noise_mu = 0;
noise_sigma = 10;
threshold_iterations = 10;
theta_MCP = 5;
theta_SCAD = 5;
a = 1;
gamma_cauchy = 2;


image = imread(strcat('images/', image_name));
image = imresize(image, 1);
image = double(image);
[height, width] = size(image);

subplot(2, 2, 1);
imshow(uint8(image));
title('Original Image');

noise = normrnd(noise_mu, noise_sigma, height, width);
noisy_image = image + round(noise);

subplot(2, 2, 2);
imshow(uint8(noisy_image));
title('Noisy Image');


image_vector = reshape(noisy_image, height*width, 1);
transformed_image_vector = fft(image_vector);
A = speye(height*width);

dg_0 = @(x) (0);

obj_fn_L1 = @(x) (1/2*norm(A*x-transformed_image_vector)^2 + lambda * norm(x, 1));

x_hat = transformed_image_vector;

stop_fn = @(obj_fn)  (@(x_prev, x_curr, iteration)(stop_fn_base(obj_fn, rtol, x_hat, x_prev, x_curr, iteration)));

stop_fn_L1 = stop_fn(obj_fn_L1);

argmin_fn_soft_lambda = get_argmin_function(lambda, 'L1', 'L2', threshold_iterations);

tic
disp('Calculating solution to problem');
x_approx = ExtendedProximalDCMethod(A, transformed_image_vector, transformed_image_vector, dg_0, argmin_fn_soft_lambda, stop_fn_L1);
t = toc

x_untransformed = ifft(x_approx);
image_approx = reshape(x_untransformed, height, width);

subplot(2, 2, 3);
imshow(uint8(image_approx));
title(sprintf('Denoised Image, PSNR = %2.2f DB', psnr(uint8(image_approx), uint8(image))));