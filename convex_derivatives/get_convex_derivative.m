function [dg] = get_convex_derivative(penalty_function, lambda, a, theta_MCP, theta_SCAD, M_arctan)
% Get the derivative of the negative part of a penalty function, based on
% the penalty function

% INPUTS:
%   penalty_function: the name of the penalty function, can be one of
%   ['arctan', 'cauchy', 'L1', 'L1_L2', 'L1_double_L2', 'L1_half_L2',
%   'MCP', 'SCAD', 'TL1']
%   lambda: the penalty parameter

% OUTPUTS:
%   dg(x): A function giving a member of the subderivative of the negative part of the penalty function

dg_L2 = @(x) lambda*dg_2_norm(x);
dg_half_L2 = @(x) lambda*dg_2_norm(x)/2;
dg_double_L2 = @(x) lambda*dg_2_norm(x)*2;
dg_0 = @(x) (0);

% DC decompositions of MCP, SCAD and Transformed L1 come from 
% https://link.springer.com/article/10.1007/s10589-017-9954-1
% All three use the L1 norm as the positive convex part
dg_MCP = @(x) (lambda.*sign(x).*min(1, abs(x)/(theta_MCP*lambda)));
dg_SCAD = @(x) (sign(x).*(max(min(theta_SCAD*lambda, abs(x)) - lambda, 0))/(theta_SCAD - 1));
dg_TL1 = @(x) (sign(x).*((a+1)/(a)) - sign(x).*(a^2 + a)./((a + abs(x)).^2));

dg_cauchy = @(x) lambda*2*x;
dg_arctan = @(x) lambda*M_arctan*x;

switch (penalty_function)
    case 'arctan'
        dg = dg_arctan;
    case 'cauchy'
        dg = dg_cauchy;
    case 'L1'
        dg = dg_0;
    case 'L1-L2'
        dg = dg_L2;
    case 'L1-double L2'
        dg = dg_double_L2;
    case 'L1-half L2'
        dg = dg_half_L2;
    case 'MCP'
        dg = dg_MCP;
    case 'SCAD'
        dg = dg_SCAD;
    case 'TL1'
        dg = dg_TL1;
end
end

function [dg] = dg_2_norm(x) 
    if x == 0
        dg = 0;
    else
        dg = x ./ norm(x, 2);
    end
end
