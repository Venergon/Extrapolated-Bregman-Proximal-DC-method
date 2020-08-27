function [g] = get_penalty_function(penalty_function, dimension, lambda, a, theta_MCP, theta_SCAD, alpha_arctan, beta_arctan, gamma_arctan, gamma_cauchy)
% Get the penalty function, based on the name

% INPUTS:
%   penalty_function: the name of the penalty function, can be one of
%   ['arctan', 'cauchy', 'L1', 'L1_L2', 'L1_double_L2', 'L1_half_L2',
%   'MCP', 'SCAD', 'TL1']
%
%   dimension: The dimension of the object to be optimized. Can be one of
%   ['1D', '2D']

%   lambda: the penalty parameter


% OUTPUTS:
%   g(x): The penalty function

switch (dimension)
    case '1D'
        switch (penalty_function)
            case 'arctan'
                g = @(x) penalty_1D_arctan(x, lambda, alpha_arctan, beta_arctan, gamma_arctan);
            case 'cauchy'
                g = @(x) penalty_1D_cauchy(x, lambda, gamma_cauchy);
            case 'L1'
                g = @(x) penalty_1D_L1(x, lambda);
            case 'L1-L2'
                g = @(x) penalty_1D_L1_L2(x, lambda, 1);
            case 'L1-double L2'
                g = @(x) penalty_1D_L1_L2(x, lambda, 2);
            case 'L1-half L2'
                g = @(x) penalty_1D_L1_L2(x, lambda, 0.5);
            case 'MCP'
                g = @(x) penalty_1D_MCP(x, lambda, theta_MCP);
            case 'SCAD'
                g = @(x) penalty_1D_SCAD(x, lambda, theta_SCAD);
            case 'TL1'
                g = @(x) penalty_1D_TL1(x, lambda, a);
            otherwise
                error('Must have correct penalty function type');
        end
    case '2D'
        switch (penalty_function)
            case 'arctan'
                g = @(x) penalty_2D_arctan(x, lambda, alpha_arctan, beta_arctan, gamma_arctan);
            case 'cauchy'
                g = @(x) penalty_2D_cauchy(x, lambda, gamma_cauchy);
            case 'L1'
                g = @(x) penalty_2D_abs(x, lambda);
            case 'L1-L2'
                g = @(x) penalty_2D_abs_frobenius(x, lambda, 1);
            case 'L1-double L2'
                g = @(x) penalty_2D_abs_frobenius(x, lambda, 2);
            case 'L1-half L2'
                g = @(x) penalty_2D_abs_frobenius(x, lambda, 0.5);
            case 'MCP'
                g = @(x) penalty_2D_MCP(x, lambda, theta_MCP);
            case 'SCAD'
                g = @(x) penalty_2D_SCAD(x, lambda, theta_SCAD);
            case 'TL1'
                g = @(x) penalty_2D_TL1(x, lambda, a);
            otherwise
                error('Must have correct penalty function type');
        end
    otherwise
        error('Dimension must be one of ["1D", "2D"]');
end
