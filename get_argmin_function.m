function [f] = get_argmin_function(lambda, penalty_type, distance_type, iterations)
% Returns a function to calculate the argmin{lambda*g_1(x) + <df(w_k) - xi, x-w_k>
% + L/2 ||x - w_k||_2^2 + 1/t*D_h(x, x^k)

% INPUTS:
% lambda: The multiplier for the penalty
% penalty_type: A string indicating the convex part of the penalty function. Can be any of ['L1']
% distance_type: A string indicating which type of distance function to
% use. Can be any of ['L2']
% iterations: the total number of iterations to use, for an iterative
% function


% Gradient of the distance operator D
%USE_KULLBACK_LEIBLER_DIVERGENCE = false;
%if USE_KULLBACK_LEIBLER_DIVERGENCE
    % TODO: This breaks if x or y are <= 0, find out what needs to be done
    % in those cases
%    dD = @(x, y) (log(x./y) + 1);
%else
%    dD = @(x, y) (x - y);
%end

% OUTPUTS:
% f(A, b, w, xi, L, t, x_prev, max_eigval): The function to
% calculate the above argmin
switch distance_type
    case 'L2'
        D = @(x, y) (1/2*norm(x - y, 2)^2);
        dD = @(x, y) (x-y);
        
    otherwise
        disp('Please select a valid distance type from the following: ["L2"]')
end


switch penalty_type
    case 'L1'
        f = @(A, b, w, xi, L, t, x_prev, max_eigval) argmin_soft_threshold(A, b, dD, w, xi, L, t, x_prev, lambda, max_eigval, iterations);
    case 'cauchy'
        gamma = 2;
        f = @(A, b, w, xi, L, t, x_prev, max_eigval) argmin_cauchy(A, b, dD, w, xi, L, t, x_prev, lambda, max_eigval, iterations, gamma);
    case 'arctan'
        beta = sqrt(3)/3;
        gamma = pi/6;
        alpha = 1;
        f = @(A, b, w, xi, L, t, x_prev, max_eigval) argmin_arctan(A, b, dD, w, xi, L, t, x_prev, lambda, max_eigval, iterations, alpha, beta, gamma);
    otherwise
        disp('Please select a valid penalty type from the following: ["L1", "cauchy"]')

end

end