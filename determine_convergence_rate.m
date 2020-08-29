lambda = 2e-5;
threshold_iterations = 100;
rtol = 1e-4;
max_iter = 200;

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


options.ti = 0;
Jmin = 4;

w= @(f) perform_wavelet_transf(f,Jmin,+1,options);
wi= @(f) perform_wavelet_transf(f,Jmin,-1,options);
%w = @(f) f;
%wi = @(f) f;


obj_fn_L1 = @(x) (f(x) + penalty_2D_abs(w(x), lambda));
obj_fn_L1_L2 = @(x) (f(x) + penalty_2D_abs_frobenius(w(x), lambda));

dg_0 = @(x) (0);
dg_L2 = get_convex_derivative('L1-L2', lambda, 0, 0, 0, 0);

dg = dg_0;
obj_fn = obj_fn_L1;
penalty_function_name = 'L1';

argmin_fn = get_argmin_function(lambda, 'L1-f', 'L2', threshold_iterations, 0, 0, 0, w, wi);

stop_fn_first = @(x_prev, x_curr, iteration)((iteration > max_iter*1.1) || stop_fn_base(obj_fn, rtol, x0, x_prev, x_curr, iteration));

tic
fprintf('Calculating once to get the optimal solution\n');
x_optimal = ExtendedProximalDCMethod(f, df, L, x0, dg, argmin_fn, stop_fn_first);
toc

stop_fn_second = @(x_prev, x_curr, iteration)(stop_fn_with_obj_value(obj_fn, rtol, x0, x_optimal, x_prev, x_curr, iteration, max_iter));

tic
fprintf('Calculating objective values at each iteration for %s penalty\n', penalty_function_name);
x_approx = ExtendedProximalDCMethod(f, df, L, x0, dg, argmin_fn, stop_fn_second);
t = toc

function [stop] = stop_fn_with_obj_value(obj_fn, rtol, x0, x_hat, x_prev, x_curr, iteration, max_iter)
    persistent obj_values;
    
    obj = obj_fn(x_curr);
    obj_prev = obj_fn(x_prev);
    obj_hat = obj_fn(x_hat);
    
    obj_err = abs(obj - obj_hat);
    obj_values(iteration+1) = obj_err;
    stop = stop_fn_base(obj_fn, rtol, x0, x_prev, x_curr, iteration);
    
    if iteration > max_iter
        stop = 1;
    end
    if stop
        iterations = 2:iteration;
        %obj_values = obj_values(2:iteration);
        
        obj_diff = log(obj_values(2:iteration)) - log(obj_values(1:iteration-1));
        iter_diff = log(2:iteration) - log(1:iteration-1);
        obj_result = log(obj_values(2:iteration)) ./ log(iterations);%obj_diff; % ./ iter_diff;

        close all;
        figure();
        plot(iterations, obj_result);
    end
end

