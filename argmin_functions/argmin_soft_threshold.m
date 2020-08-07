function [x] = argmin_soft_threshold(A, b, dD, w, xi, L, t, x_prev, lambda, max_eigval, thresholding_iterations)
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
df = (ifft(w) - b);%A'*(A*w - b);

n = length(x_prev);

%if norm(x_prev, 1) <= 826292734.775
%    step_size = 0.9/(2*max_eigval*length(x_prev));
%else
    step_size = 0.9/(2*(L + 1/t));
%end
%step_size = 1/100000;

dh = @(x) (df - xi + L.*(x-w) + (1/t).*dD(x, x_prev));

x = x_prev;

obj_fn = @(x) (lambda*norm(x, 1) + sum((df - xi).*(x-w)) + L/2*(norm(x-w, 2)^2) + (1/t) * (1/2) * (norm(x-x_prev, 2)^2));


for iteration = 1:thresholding_iterations
    inner_vector = x - step_size*dh(x);

    for i=1:n
        x(i) = sign(inner_vector(i))*max((abs(inner_vector(i)) - lambda*step_size), 0);
    end
end

%if obj_fn(x) > obj_fn(x_prev)
    %x = x_prev;
    %fprintf("Wat\n\n");
    
    %f = @(x) (lambda*abs(x) + (df(i) - xi)*(x-w(i)) + L/2*(x - w(i))^2 + 1/(2*t) * (x-x_prev(i))^2);
    %for i=1:n
    %    if f(x(i)) > f(x_prev(i))
    %        fprintf("HUH?\n");
    %        f(x(i))
    %        f(x_prev(i))
    %        f(x(i)) - f(x_prev(i))
    %        i
    %        x(i) - x_prev(i)
    %        x(i)
    %        x_prev(i)
    %        error("test");
    %    end
    %end
    
%end
end

        
