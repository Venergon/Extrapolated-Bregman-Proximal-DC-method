function [f, df, L] = get_objective_function(objective_type, A, b)
% Gets the objective function, derivative and lipschitz constant of df
% Inputs:
%   objective_type: the type of objective function used. Can be any of
%       ['1D-L2', '2D-fro']
%   A: The matrix, for matrix based objective functions
%   b: The desired result of applying the matrix to the current value, for
%       matrix based objective functions


switch objective_type
    case '1D-L2'
        f = @(x) objective_1D_L2(A, x, b);
        df = @(x) objective_derivative_1D_L2(A, x, b);
        L = objective_lipschitz_1D_L2(A, b);
        
    case '2D-fro'
        f = @(x) objective_2D_frobenius(A, x, b);
        df = @(x) objective_derivative_2D_frobenius(A, x, b);
        L = objective_lipschitz_2D_frobenius(A, b);
        
    otherwise
        error("Objective function must be one of ['1D-L2', '2D-fro']");
end

