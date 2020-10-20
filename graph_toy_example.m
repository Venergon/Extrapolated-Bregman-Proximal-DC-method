% Graph the solutions to the 'toy' example in https://arxiv.org/pdf/1812.08852.pdf
% With various penalty functions and initial points

rtol = 1e-10;
lambda = 0.1;

beta_arctan = sqrt(3)/3;
gamma_arctan = pi/6;
alpha_arctan = 1;
a = 1;
gamma_cauchy = 2;
threshold_iterations = 10;
theta_MCP = 5;
theta_SCAD = 5;



M_arctan = (2*alpha_arctan^2*beta_arctan)/(gamma_arctan*(1+beta_arctan^2));

M_cauchy = 2;

A = [1, -1, 0, 0, 0, 0;
    1, 0, -1, 0, 0, 0;
    0, 1, 1, 1, 0, 0;
    2, 2, 0, 0, 1, 0;
    1, 1, 0, 0, 0, -1];

x_ideal = [0; 0; 0; 20; 40; -18];
b = A*x_ideal;

[f, df, L] = get_objective_function('1D-L2', A, b);

close all;
t_fig = figure();
hold on;
diff_fig = figure();
hold on;

penalty_functions = ["arctan", "cauchy", "L1", "L1-L2", "L1-double L2", "L1-half L2", "MCP", "SCAD", "TL1"];

for penalty_function_no = 1:length(penalty_functions)
    penalty_function_name = penalty_functions(penalty_function_no);
    fprintf("Starting on %s penalty\n", penalty_function_name);

    g = get_penalty_function(penalty_function_name, '1D', lambda, a, theta_MCP, theta_SCAD, alpha_arctan, beta_arctan, gamma_arctan, gamma_cauchy);
    dg = get_convex_derivative(penalty_function_name, lambda, a, theta_MCP, theta_SCAD, M_arctan);
    argmin_fn = get_argmin_fn_for_penalty(penalty_function_name, lambda, threshold_iterations, a, alpha_arctan, beta_arctan, gamma_arctan, gamma_cauchy);

    obj_fn = @(x) (f(x) + g(x));

    t_vals = -20:20;
    approx_t_vals = zeros(size(t_vals));
    diff_vals = zeros(size(t_vals)); 

    for i = 1:length(t_vals)
        t = t_vals(i);
        x0 = get_toy_sol_for_param(t);
        
        stop_fn = @(x_prev, x_curr, iteration)(stop_fn_base(obj_fn, rtol, x0, x_prev, x_curr, iteration));
        x_approx = ExtrapolatedProximalDCMethod(f, df, L, x0, dg, argmin_fn, stop_fn);
        b_approx = A*x_approx;

        obj_ideal = obj_fn(x_ideal);
        obj_approx = obj_fn(x_approx);

        [approx_t, diff] = find_closest_toy_parameter(x_approx);
        approx_t_vals(i) = approx_t;
        diff_vals(i) = diff;
    end
    
    figure(t_fig);
    plot(t_vals, approx_t_vals, '-x', 'DisplayName', penalty_function_name);
    
    figure(diff_fig);
    plot(t_vals, diff_vals, '-x', 'DisplayName', penalty_function_name);
end

figure(t_fig);
yline(0, 'DisplayName', 'Optimal Solution');
xlabel('t for x0');
ylabel('approx t for solution');

legend('Location', 'SouthWest');

figure(diff_fig);
xlabel('t for x0');
ylabel('diff from closest t for solution');

legend('Location', 'NorthWest');

function [t, diff] = find_closest_toy_parameter(x)
    % Find the closest paramter that matches the solution we have.
    
    t_matrix = [1, 1, 1, -2, -4, 2]';
    
    t = t_matrix \ (x - [0, 0, 0, 20, 40, -18]');
    
    sol = get_toy_sol_for_param(t);
    
    diff = norm(sol - x, 2);
end

function [x] = get_toy_sol_for_param(t)
    x = [t, t, t, 20 - 2*t, 40 - 4*t, 2*(t - 9)]';
end