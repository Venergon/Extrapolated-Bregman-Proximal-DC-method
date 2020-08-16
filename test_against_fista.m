X = double(imread('images/cameraman.pgm'));
X = X/255;
[P, center] = psfGauss([9, 9], 4);

B = imfilter(X, P, 'symmetric');

randn('seed', 314);
Bobs = B + 1e-3*randn(size(B));

subplot(2,2,1);
imshow(X,[]);
title('Original Image');
subplot(2,2,2);
imshow(Bobs,[]);
title('Blurred Image');

fpic = @(x) norm(imfilter(x,P,'symmetric') - Bobs,'fro')^2;
grad_fpic = @(x) 2* imfilter(imfilter(x,P,'symmetric')-Bobs,P,'symmetric');
f = @(x) fpic(x);
df = @(x) grad_fpic(x);
L = 1;

x_hat = Bobs;
x0 = Bobs;

obj_fn_L1 = @(x) (f(x) + penalty_2D_abs(x, lambda));

lambda = 2e-5;
threshold_iterations = 10;
rtol = 1e-4;

stop_fn = @(obj_fn)  (@(x_prev, x_curr, iteration)(stop_fn_base(obj_fn, rtol, x_hat, x_prev, x_curr, iteration)));

dg_0 = @(x) (0);

stop_fn_L1 = stop_fn(obj_fn_L1);

argmin_fn_soft_lambda = get_argmin_function(lambda, 'L1', 'L2', threshold_iterations);


tic
disp('Calculating solution to problem');
x_approx = ExtendedProximalDCMethod(f, df, L, x0, dg_0, argmin_fn_soft_lambda, stop_fn_L1);
t = toc


subplot(2,2,3);
imshow(x_approx,[]);
title('Recovered Image');