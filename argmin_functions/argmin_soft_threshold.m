function [x] = argmin_soft_threshold(df, dD, w, xi, L, t, x_prev, lambda, thresholding_iterations)
% Calculates a value x \in argmin_{x} {lambda*|x|_1 + <\nabla f(w_k) - xi_k, x - w_k> +
% L/2*||x - w_k||_2^2 + 1/t_k*(D(x - x_curr))
% Where f(x) = 1/2 * ||Ax - b||^2
% And D is a distance function (currently ||x - y||_2 distance)

% Solve using soft thresholding from https://angms.science/doc/CVX/ISTA0.pdf
% Let h(x) = <\nabla f(w_k) - xi_k, x - w_k> +
% L/2*||x - w_k||_2^2 + 1/t_k*(1/2)*(||x - x_curr||_2)^2

% And create a new vector x, where for each index i x_i = sgn([x_curr -
% \nabla h(x_curr)]_i)(|[x_curr - \nabla h(x_curr)]_i - lambda)

% Represent \nabla f and \nabla h by df and dh respectively
df_w = df(w);

step_size = 0.9/(2*(lambda + L + 1/t));

dh = @(x) (df_w - xi + L.*(x-w) + (1/t).*dD(x, x_prev));

x = x_prev;


for iteration = 1:thresholding_iterations
    inner_vector = x - step_size*dh(x);

    x = sign(inner_vector).*max((abs(inner_vector) - lambda*step_size), 0);
end

obj_fn = @(x) (lambda*sum(abs(x), 'all') + trace((df_w - xi)'*(x-w)) + L/2*(norm(x-w, 'fro')^2) + (1/t) * (1/2) * (norm(x-x_prev, 'fro')^2));

if obj_fn(x) > obj_fn(x_prev)
    fprintf("Error: SOFT THRESHOLDING ended with higher objective value\n\n");
end

end

        
