USE_2_NORM = true;
tol = 1e-3;
lambda = 0.01;


A = [1, -1, 0, 0, 0, 0;
    1, 0, -1, 0, 0, 0;
    0, 1, 1, 1, 0, 0;
    2, 2, 0, 0, 1, 0;
    1, 1, 0, 0, 0, -1];

x_ideal = [0; 0; 0; 20; 40; -18];
b = A*x_ideal;

x0 = [1; 1; 1; 1; 1; 1];
% g = ||x||_2, so dg = 


if USE_2_NORM
    dg = @(x) (x./norm(x, 2));
    obj_fn = @(x) (norm(A*x-b)^2 + lambda *(norm(x, 1) - norm(x, 2)));
else
    dg = @(x) (0);
    obj_fn = @(x) (norm(A*x-b)^2 + lambda * norm(x, 1));
end
    

stop_fn = @(x_prev, x_curr)(obj_fn(x_curr) > obj_fn(x_prev) && obj_fn(x_curr) - obj_fn(x_prev) < tol);



x_approx = ExtendedProximalDCMethod(A, b, x0, dg, lambda, stop_fn);
b_approx = A*x_approx;

obj_ideal = obj_fn(x_ideal);
obj_approx = obj_fn(x_approx);