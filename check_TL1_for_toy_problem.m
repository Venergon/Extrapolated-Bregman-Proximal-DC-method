% Graph the TL1 penalty for the exact solutions to the toy problem
lambda = 0.01;

beta_arctan = sqrt(3)/3;
gamma_arctan = pi/6;
alpha_arctan = 1;
a = 1;
gamma_cauchy = 2;
threshold_iterations = 10;
theta_MCP = 5;
theta_SCAD = 5;


t = -100:0.1:100;
y = zeros(size(t));

g = get_penalty_function('TL1', '1D', lambda, a, theta_MCP, theta_SCAD, alpha_arctan, beta_arctan, gamma_arctan, gamma_cauchy);


for i = 1:length(t)
    solution = get_toy_sol_for_param(t(i));
    y(i) = g(solution);
end

plot(t, y);

function [x] = get_toy_sol_for_param(t)
    x = [t, t, t, 20 - 2*t, 40 - 4*t, 2*(t - 9)]';
end