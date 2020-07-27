function [x] = argmin_soft_threshold(A, b, dD, w, xi, L, t, x_prev, lambda, max_eigval, thresholding_iterations)
% Calculates a value x \in argmin_{x} {lambda*|x|_1 + <\nabla f(w_k) - xi_k, x - w_k> +
% L/2*||x - w_k||_2^2 + 1/t_k*(1/2)*(D(x - x_curr))^2
% Where f(x) = 1/2 * ||Ax - b||^2
% And D is a distance function (currently ||x - y||_2 distance)

% Solve using soft thresholding from https://angms.science/doc/CVX/ISTA0.pdf
% Let h(x) = <\nabla f(w_k) - xi_k, x - w_k> +
% L/2*||x - w_k||_2^2 + 1/t_k*(1/2)*(||x - x_curr||_2)^2

% And create a new vector x, where for each index i x_i = sgn([x_curr -
% \nabla h(x_curr)]_i)(|[x_curr - \nabla h(x_curr)]_i - lambda)

% Represent \nabla f and \nabla h by df and dh respectively
df = A'*(A*w - b);

n = length(x_prev);
step_size = 1/(2*max_eigval);

dh = @(x) (df - xi.*ones(n, 1) + L.*(x-w) + (1/t).*dD(x, x_prev));

x = x_prev;

for iteration = 1:thresholding_iterations
    inner_vector = x_prev - step_size*dh(x);

    for i=1:n
        x(i) = sign(inner_vector(i))*max((abs(inner_vector(i)) - lambda*step_size), 0);
    end
end

end