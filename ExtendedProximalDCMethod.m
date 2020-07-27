function [x] = ExtendedProximalDCMethod(A, b, x0, dg_2, calculate_argmin, stop_fn)
% Extended Proximal DC Method: Find an approximation to the minimum of
% 1/2*|Ax-b|^2 + lambda*(g_1(x) - g_2(x))
% using the Extended Proximal DC method
% 
% Inputs:
%  A : matrix used for the minimization of |Ax-b|
%  b : desired result vector, used in the minimization of |Ax - b|
%  x0 : initial guess for x
%  dg_2(x) : a function that returns a member of the subderivative of g_2 at x 
%  calculate_argmin(A, b, w, xi, L, t, x_curr, max_eigval) : the
%  function used to get argmin{1/2*|Ax-b|^2 + lambda*g_1(x) +
%  L/2*||x-w^k||_2^2 + 1/t_k*D_h(x, x_k)}
%  stop_fn(x_prev, x_curr, iteration) : The condition on x_k and x_{k-1} under which to stop
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

max_eigval = eigs(A'*A, 1);

% Calculate L as a lipschitz constant for the gradient of 1/2*|Ax - b|^2
L = max_eigval;

% Set the parameters for the extapolation parameter, based on the method to pick kappa in 
% remark 3.2 in https://arxiv.org/pdf/2003.04124.pdf
% Choose alpha_max as close as possible to 1 without reaching 1
alpha_max = 0.99;
nu_curr = 1;
n0 = 10000;
max_diff = 0;

first_iteration = true;
% Iterate until the stopping condition is reached, ignoring the first
% iteration
iteration = 0;
while ~isnan(x_curr(1)) && ((first_iteration) || (~stop_fn(x_prev, x_curr, iteration)))
    iteration = iteration + 1;
    first_iteration = false;
    
    if (mod(iteration, 1000) == 0)
        fprintf('iteration %d\n', iteration);
    end
    
    %% Step 1a: Choose alpha_k >= 0 and compute w_k = x_k + alpha_k*(x_k -
    % x_{k-1})
    if (mod(iteration, n0) == 0)
        % Reset nu every n0 iterations
        nu_prev = 1;
        nu_curr = 1;
    else
        nu_prev = nu_curr;
        nu_curr = (1 + sqrt(1 + 4*nu_prev^2))/2;
    end
    
    alpha = alpha_max * (nu_prev - 1)/nu_curr;
    %alpha = 1;
    
    w = x_curr + alpha.*(x_curr - x_prev);
    
    %% Step 1b: Compute xi_k \in \partial{lambda * g}(x_k)
    % lambda already comes from dg_2
    xi = dg_2(x_curr);
    
    %% Step 2: Compute the step size t_k and  update x_{k+1} by letting
    % x_{k+1} \in argmin_{x} {lambda*|x|_1 + <\nabla f(w_k) - xi_k, x - w_k> +
    % L/2*||x - w_k||_2^2 + 1/t_k*D(x, x_k)
    
    % TODO: Currently got a static step size t_k = 1
    t = 100;
    
    x_next = calculate_argmin(A, b, w, xi, L, t, x_curr, max_eigval);
   
    %% Shuffle x_prev, x_curr, x_next to represent moving to the next value of k
    x_prev = x_curr;
    x_curr = x_next;
    
    diff = norm(x_prev - x_curr);
    if diff > max_diff
        max_diff = diff;
    end
end

x = x_curr;

fprintf('Finished in %d iterations\n', iteration);
end

