% Generate blurred image with random noise, and solve using L_{1, 1}, L_{1,
% 1} - L_{2, 2}, ISTA and FISTA algorithms
% Display all solutions as images

lambda = 2e-5;
threshold_iterations = 100;
rtol = 1e-4;
max_iter = 200;
weighting = 9000;


close all;

X = double(imread('images/cameraman.pgm'));
X = X/255;
[P, center] = psfGauss([9, 9], 4);

B = imfilter(X, P, 'symmetric');

randn('seed', 314);
Bobs = B + 1e-3*randn(size(B));

figure();
subplot(1, 2, 1);
imshow(X,[]);
title('Original Image', 'interpreter', 'latex');
subplot(1, 2, 2);
imshow(Bobs,[]);
title(sprintf('Blurred Image,\nPSNR = %2.2f dB', psnr(Bobs, X)), 'interpreter', 'latex');

[f, df, L] = get_objective_function('2D-filter', 0, Bobs, P);

x_hat = Bobs;
x0 = Bobs;


options.ti = 0;
Jmin = 4;

w= @(f) perform_wavelet_transf(f,Jmin,+1,options);
wi= @(f) perform_wavelet_transf(f,Jmin,-1,options);

obj_fn_L1 = @(x) (f(x) + penalty_2D_abs(w(x), lambda));
obj_fn_L1_L2 = @(x) (f(x) + penalty_2D_abs_frobenius(w(x), lambda, weighting));


stop_fn = @(obj_fn)  (@(x_prev, x_curr, iteration)(iteration == max_iter));

dg_0 = @(x) (0);
dg_fro = get_convex_derivative('L1-fro', lambda, 0, 0, 0, 0);
dg_L2 = @(x) (weighting*dg_fro(x));

stop_fn_L1 = stop_fn(obj_fn_L1);
stop_fn_L1_L2 = stop_fn(obj_fn_L1_L2);

argmin_fn_soft_lambda = get_argmin_function(lambda, 'L1-f', 'L2', threshold_iterations, 0, 0, 0, w, wi);

tic
disp('Calculating L1 solution to problem');
x_L1 = ExtrapolatedProximalDCMethod(f, df, L, x0, dg_0, argmin_fn_soft_lambda, stop_fn_L1);
t_L1 = toc


figure();
subplot(1, 2, 1);
imshow(x_L1,[]);
title(sprintf('$L_{1,1}$ penalty (%d iterations),\nPSNR = %2.2f dB', max_iter, psnr(x_L1, X)), 'interpreter', 'latex');

tic
disp('Calculating L1-L2 solution to problem');
x_L1_L2 = ExtrapolatedProximalDCMethod(f, df, L, x0, dg_L2, argmin_fn_soft_lambda, stop_fn_L1_L2);
t_L1_L2 = toc

subplot(1, 2, 2);
imshow(x_L1_L2,[]);
title(sprintf('$L_{1,1}-%dL_{2,2}$ penalty (%d iterations),\nPSNR = %2.2f dB', weighting, max_iter, psnr(x_L1_L2, X)), 'interpreter', 'latex');


Gpic = @(x)  sum(sum(abs(w(x))));
prox_gpic = @(x,a) wi(prox_l1(w(x),a));

clear par;
par.max_iter = max_iter;
x_pg=prox_gradient(f,df, @(x) Gpic(x), @(x,alpha)prox_gpic(x,alpha),lambda,x0,par);

par.max_iter = max_iter - 1;
x_pg_less = prox_gradient(f,df, @(x) Gpic(x), @(x,alpha)prox_gpic(x,alpha),lambda,x0,par);

par.max_iter = max_iter + 1;
x_pg_more = prox_gradient(f,df, @(x) Gpic(x), @(x,alpha)prox_gpic(x,alpha),lambda,x0,par);

figure()
subplot(1, 2, 1);
imshow(x_pg, []);
title(sprintf('ISTA (%d iterations),\nPSNR = %2.2f dB', max_iter, psnr(x_pg, X)), 'interpreter', 'latex');

x_fista = fista(f, df, @(x) Gpic(x), @(x, alpha) prox_gpic(x, alpha), lambda, x0, par);

subplot(1, 2, 2);
imshow(x_fista, []);
title(sprintf('FISTA (%d iterations),\nPSNR = %2.2f dB', max_iter, psnr(x_fista, X)), 'interpreter', 'latex');


figure()
subplot(1, 2, 1);
imshow(x_pg, []);
title(sprintf('ISTA (%d iterations),\nPSNR = %2.2f dB', max_iter, psnr(x_pg, X)), 'interpreter', 'latex');

subplot(1, 2, 2);
imshow(x_L1,[]);
title(sprintf('$L_{1,1}$ penalty (%d iterations),\nPSNR = %2.2f dB', max_iter, psnr(x_L1, X)), 'interpreter', 'latex');

figure()
subplot(1, 2, 1);
imshow(x_fista, []);
title(sprintf('FISTA (%d iterations),\nPSNR = %2.2f dB', max_iter, psnr(x_fista, X)), 'interpreter', 'latex');

subplot(1, 2, 2);
imshow(x_L1,[]);
title(sprintf('$L_{1,1}$ penalty (%d iterations),\nPSNR = %2.2f dB', max_iter, psnr(x_L1, X)), 'interpreter', 'latex');

diff_less = norm(x_pg_less - x_L1, inf)
diff = norm(x_pg - x_L1, inf)
diff_more = norm(x_pg_more - x_L1, inf)