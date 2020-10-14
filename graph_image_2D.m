close all;

lambda = 2e-5;
threshold_iterations = 100;
rtol = 1e-4;
max_iter = 2000;

weightings = 0:1000:1e5;%[1e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 6e4, 7e4, 1e5];
psnrs = zeros(size(weightings));


close all;

X = double(imread('images/cameraman.pgm'));
X = X/255;
[P, center] = psfGauss([9, 9], 4);

B = imfilter(X, P, 'symmetric');

randn('seed', 314);
Bobs = B + 1e-3*randn(size(B));

[f, df, L] = get_objective_function('2D-filter', 0, Bobs, P);

x_hat = Bobs;
x0 = Bobs;


options.ti = 0;
Jmin = 4;

w= @(f) perform_wavelet_transf(f,Jmin,+1,options);
wi= @(f) perform_wavelet_transf(f,Jmin,-1,options);



stop_fn = @(obj_fn)  (@(x_prev, x_curr, iteration)((iteration == max_iter) || (norm(x_prev - x_curr, 'inf') < 0.01)));

dg_0 = @(x) (0);
dg_fro = get_convex_derivative('L1-fro', lambda, 0, 0, 0, 0);


argmin_fn_soft_lambda = get_argmin_function(lambda, 'L1-f', 'L2', threshold_iterations, 0, 0, 0, w, wi);


for i=1:length(weightings)
    weighting = weightings(i);
    
    dg_L2 = @(x) (weighting*dg_fro(x));
    obj_fn_L1_L2 = @(x) (f(x) + penalty_2D_abs_frobenius(w(x), lambda, weighting));
    stop_fn_L1_L2 = stop_fn(obj_fn_L1_L2);

    tic
    fprintf('Calculating L1- %f L2 solution to problem\n', weighting);
    x_L1_L2 = ExtrapolatedProximalDCMethod(f, df, L, x0, dg_L2, argmin_fn_soft_lambda, stop_fn_L1_L2);
    t_L1_L2 = toc

    psnrs(i) = psnr(x_L1_L2, X);
end

plot(weightings, psnrs);