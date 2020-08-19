lambda = 2e-5;
threshold_iterations = 100;
rtol = 1e-4;
max_iter = 100;

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

[f, df, L] = get_objective_function('2D-filter', 0, Bobs, P);

x_hat = Bobs;
x0 = Bobs;

obj_fn_L1 = @(x) (f(x) + penalty_2D_abs(x, lambda));


stop_fn = @(obj_fn)  (@(x_prev, x_curr, iteration)((iteration == max_iter) || (0 && stop_fn_base(obj_fn, rtol, x_hat, x_prev, x_curr, iteration))));

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

options.ti = 0;
Jmin = 4;
%w= @(f) perform_wavelet_transf(f,Jmin,+1,options);
%wi= @(f) perform_wavelet_transf(f,Jmin,-1,options);
w = @(f) f;
wi = @(f) f;

Gpic = @(x)  sum(sum(abs(w(x))));
prox_gpic = @(x,a) wi(prox_l1(w(x),a));

clear par;
par.max_iter = max_iter;
x_pg=prox_gradient(f,df, @(x) Gpic(x), @(x,alpha)prox_gpic(x,alpha),lambda,x0,par);

subplot(2, 2, 4);
imshow(x_pg, []);
title('proximal gradient');

diff = norm(x_pg - x_approx, inf)