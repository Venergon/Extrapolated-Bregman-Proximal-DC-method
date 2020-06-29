function [x] = calculate_argmin(A, b, w, xi, L, t, x_curr, lambda)
% Calculates a value x \in argmin_{x} {lambda*|x|_1 + <\nabla f(w_k) - xi_k, x - w_k> +
% L/2*||x - w_k||_2^2 + 1/t_k*(1/2)*(||x - x_curr||_2)^2
% Where f(x) = 1/2 * ||Ax - b||^2

% Solve using soft thresholding from https://angms.science/doc/CVX/ISTA0.pdf
% Let h(x) = <\nabla f(w_k) - xi_k, x - w_k> +
% L/2*||x - w_k||_2^2 + 1/t_k*(1/2)*(||x - x_curr||_2)^2

% And create a new vector x, where for each index i x_i = sgn([x_curr -
% \nabla h(x_curr)]_i)(|[x_curr - \nabla h(x_curr)]_i - lambda)

% Represent \nabla f and \nabla h by df and dh respectively
df = A'*(A*w - b);

step_size = 0.05;
n = length(x_curr);
dh = @(x) (lambda*sign(x) + df - xi.*ones(n, 1) + L.*(x-w) + (1/t).*(x-x_curr));

x = zeros(n, 1);

w
xi
df
dh_x = dh(x_curr)
inner_vector = x_curr - step_size*dh(x_curr);

for i=1:n
    x(i) = sign(inner_vector(i))*(abs(inner_vector(i)) - lambda*step_size);
end

end