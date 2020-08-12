% The 'toy' example in https://arxiv.org/pdf/1812.08852.pdf

tol = 1e-10;
lambda = 1;

beta_arctan = sqrt(3)/3;
gamma_arctan = pi/6;
alpha_arctan = 1;

M_arctan = (2*alpha_arctan^2*beta_arctan)/(gamma_arctan*(1+beta_arctan^2));

M_cauchy = 2;

A = [1, -1, 0, 0, 0, 0;
    1, 0, -1, 0, 0, 0;
    0, 1, 1, 1, 0, 0;
    2, 2, 0, 0, 1, 0;
    1, 1, 0, 0, 0, -1];

x_ideal = [0; 0; 0; 20; 40; -18];
b = A*x_ideal;


% Because there's no noise, the least squares solution will give an exact
% solution to the ||Ax - b|| = 0 but not necessarily the solution to the
% penalty problem (running it locally gives me ~[10;10;10;0;0;2], which is
% not optimal. Starting from a zero vector also results in a solution very
% close to the optimum, so either work as a starting guess.

% A choice that means ||Ax-b|| = 0, but is closer to the actual solution
% than the local minimizer for many of the penalty functions (~[9.5, 10,
% 9.5, 0, 0, 0.5])
x0 = [-1, -1, -1, 22, 44, -10]';%A \ b;%zeros(6, 1);

dg_L2 = @(x) lambda*dg_2_norm(x);
dg_0 = @(x) (0);

% DC decompositions of MCP, SCAD and Transformed L1 come from 
% https://link.springer.com/article/10.1007/s10589-017-9954-1
% All three use the L1 norm as the positive convex part
theta_MCP = 5;
dg_MCP = @(x) (lambda.*sign(x).*min(1, abs(x)/(theta_MCP*lambda)));
theta_SCAD = 5;
dg_SCAD = @(x) (sign(x).*(min(theta_SCAD*lambda, abs(x)) - lambda)/(theta_SCAD - 1));
a = 1;
dg_TL1 = @(x) (sign(x).*((a+1)/(a)) - sign(x).*(a^2 + a)./((a + abs(x)).^2));

gamma_cauchy = 2;
dg_cauchy = @(x) lambda*M_cauchy*x;
dg_arctan = @(x) lambda*M_arctan*x;


dg = dg_cauchy;
obj_fn = @(x) (objective_1D_L2(A, x, b) + penalty_1D_cauchy(x, lambda, gamma_cauchy));

stop_fn = @(x_prev, x_curr, iteration)(obj_fn(x_curr) < obj_fn(x_prev) && obj_fn(x_prev) - obj_fn(x_curr) < tol);


threshold_iterations = 10;
argmin_function = get_argmin_function(lambda, 'cauchy', 'L2', threshold_iterations);
x_approx = ExtendedProximalDCMethod(A, b, x0, dg, argmin_function, stop_fn);
b_approx = A*x_approx;

obj_ideal = obj_fn(x_ideal);
obj_approx = obj_fn(x_approx);

function [dg] = dg_2_norm(x) 
    if x == 0
        dg = 0;
    else
        dg = x ./ norm(x, 2);
    end
end