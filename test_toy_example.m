USE_2_NORM = false;
tol = 1e-8;
lambda = 0.1;


A = [1, -1, 0, 0, 0, 0;
    1, 0, -1, 0, 0, 0;
    0, 1, 1, 1, 0, 0;
    2, 2, 0, 0, 1, 0;
    1, 1, 0, 0, 0, -1];

x_ideal = [0; 0; 0; 20; 40; -18];
b = A*x_ideal;

% Because there's no noise, the least squares solution will give an exact
% solution to the ||Ax - b|| = 0 but not necessarily the solution to the
% penalty problem (running it locally gives me ~[10;10;10;0;0;2], which is
% not optimal. Starting from a zero vector also results in a solution very
% close to the optimum, so either work as a starting guess.
x0 = A \ b;%zeros(6, 1);

if USE_2_NORM
    dg = @(x) (dg_2_norm(x));
    obj_fn = @(x) (norm(A*x-b)^2 + lambda *(norm(x, 1) - norm(x, 2)));
else
    dg = @(x) (0);
    obj_fn = @(x) (norm(A*x-b)^2 + lambda * norm(x, 1));
end
    

stop_fn = @(x_prev, x_curr, iteration)(obj_fn(x_curr) < obj_fn(x_prev) && obj_fn(x_prev) - obj_fn(x_curr) < tol);


threshold_iterations = 10;
argmin_function = get_argmin_function(lambda, 'L1', 'L2', threshold_iterations);
x_approx = ExtendedProximalDCMethod(A, b, x0, dg, argmin_function, stop_fn);
b_approx = A*x_approx;

obj_ideal = obj_fn(x_ideal);
obj_approx = obj_fn(x_approx);

function [dg] = dg_2_norm(x) 
    if x == 0
        dg = 0;
    else
        dg = x ./ norm(x, 2);
    end
end