function [x] = ExtendedProximalDCMethod(A, b, x0, dg_2, lambda, threshold_iterations, stop_fn)
% Extended Proximal DC Method: Find an approximation to the minimum of
% 1/2*|Ax-b|^2 + lambda*(g_1(x) - g_2(x)), where g_1(x) = ||x||_1
% using the Extended Proximal DC method
% 
% Inputs:
%  A : matrix used for the minimization of |Ax-b|
%  b : desired result vector, used in the minimization of |Ax - b|
%  x0 : initial guess for x
%  dg_2(x) : a function that returns a member of the subderivative of g_2 at x 
%  lambda : penalty parameter
%  threshold_iterations: The number of iterations used for soft
%  thresholding
%  stop_fn(x_prev, x_curr) : The condition on x_k and x_{k-1} under which to stop
% 
% Outputs:
%  x : estimate for the minimizer for the above equation
%
% Uses Algorithm 2 in https://www.overleaf.com/project/5eeb8a66da60180001e536fe

% Initialise parameters
% x_curr = x_k (x_0 initially)
% x_prev = x_{k-1} (x_{-1}, initialised with x0)
x_curr = x0;
x_prev = x_curr;

% Calculate L as a lipschitz constant for the gradient of 1/2*|Ax - b|^2
L = norm(A'*A, 2);

obj_fn = @(x) (norm(A*x-b, 2)^2 + lambda*norm(x, 1));

max_eigval = eigs(A'*A, 1);


first_iteration = true;
% Iterate until the stopping condition is reached, ignoring the first
% iteration
iteration = 0;
while ~isnan(x_curr(1)) && ((first_iteration) || (~stop_fn(x_prev, x_curr)))
    iteration = iteration + 1;
    first_iteration = false;

    obj_difference = obj_fn(x_prev) - obj_fn(x_curr);
    
    if (obj_difference < 0) 
        fprintf('Error: obj_difference %e is negative\n', obj_difference);
        %throw(MException('TEST'));
    end
    
    if (mod(iteration, 1000) == 0)
        fprintf('Iteration: %d\n', iteration);
        fprintf('Previous 2 obj values: %e %e\n', obj_fn(x_prev), obj_fn(x_curr));
        fprintf('Diff: %e\n', obj_fn(x_prev) - obj_fn(x_curr));
    end
    
    %% Step 1a: Choose alpha_k >= 0 and compute w_k = x_k + alpha_k*(x_k -
    % x_{k-1})
    
    % TODO: temporarily choosing alpha_k = 0.5, replace with some way to
    % choose alpha_k
    alpha = 0;
    
    w = x_curr + alpha*(x_curr - x_prev);
    
    %% Step 1b: Compute xi_k \in \partial{lambda * g}(x_k)
    % lambda already comes from dg_2
    xi = dg_2(x_curr);
    
    %% Step 2: Compute the step size t_k and  update x_{k+1} by letting
    % x_{k+1} \in argmin_{x} {lambda*|x|_1 + <\nabla f(w_k) - xi_k, x - w_k> +
    % L/2*||x - w_k||_2^2 + 1/t_k*D(x, x_k)
    
    % TODO: Currently got a static step size t_k = 1
    t = 1;
    
    x_next = calculate_argmin(A, b, w, xi, L, t, x_curr, lambda, max_eigval, threshold_iterations);
   
    %% Shuffle x_prev, x_curr, x_next to represent moving to the next value of k
    x_prev = x_curr;
    x_curr = x_next;
end

x = x_curr;



fprintf('Finished in %d iterations\n', iteration);
fprintf('Previous 2 obj values: %f %f\n', obj_fn(x_prev), obj_fn(x_curr));
fprintf('Diff: %f\n', obj_fn(x_prev) - obj_fn(x_curr));
end

