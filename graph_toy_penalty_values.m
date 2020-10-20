% Graph the values of various penalty functions of the solutions to the 'toy' example in https://arxiv.org/pdf/1812.08852.pdf

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

close all;
penalty_fig = figure();
hold on;

penalty_functions = ["arctan", "cauchy", "L1", "L1-L2", "L1-double L2", "L1-half L2", "MCP", "SCAD", "TL1"];

t_vals = -2:12;

% Plot all the penalty functions
for penalty_function_no = 1:length(penalty_functions)
    penalty_function_name = penalty_functions(penalty_function_no);
    fprintf("Starting on %s penalty\n", penalty_function_name);

    g = get_penalty_function(penalty_function_name, '1D', lambda, a, theta_MCP, theta_SCAD, alpha_arctan, beta_arctan, gamma_arctan, gamma_cauchy);

    penalty_vals = zeros(size(t_vals));

    for i = 1:length(t_vals)
        t = t_vals(i);
        x0 = get_toy_sol_for_param(t);
        
        
        penalty_vals(i) = g(x0);
    end
    
    figure(penalty_fig);
    plot(t_vals, penalty_vals, '-', 'DisplayName', penalty_function_name);
end

%Plot L1 - L2
penalty_vals = zeros(size(t_vals));
for i = 1:length(t_vals)
    t = t_vals(i);
    x0 = get_toy_sol_for_param(t);
    
    penalty_vals(i) = norm(x0, 1)/norm(x0, 2);
end
plot(t_vals, penalty_vals, '--', 'DisplayName', 'L1/L2');

figure(penalty_fig);
xlabel('t for x0');
ylabel('penalty value for x0');

legend('Location', 'best');

title('Penalty Function Values for Exact Solutions to Toy Problem');

function [x] = get_toy_sol_for_param(t)
    x = [t, t, t, 20 - 2*t, 40 - 4*t, 2*(t - 9)]';
end