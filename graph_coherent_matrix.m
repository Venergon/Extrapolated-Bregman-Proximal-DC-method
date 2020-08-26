% Graph ExtendedProximalDCMethod using a highly coherent randomly generated matrix of size
% nXm, with some gaussian noise
close all;

rng_seed = 1;

rng(rng_seed);

rtol = 1e-6;
lambda = 10;
m = 1024;
matrix_noise = 0.1;
density = 0.1;
noise_mu = 0;
noise_sigma = 0.1;
threshold_iterations = 10;
theta_MCP = 5;
theta_SCAD = 5;
a = 1;
gamma_cauchy = 2;

beta_arctan = sqrt(3)/3;
gamma_arctan = pi/6;
alpha_arctan = 1;

M_arctan = (3*alpha_arctan^2*beta_arctan^(2/3))/(4*gamma_arctan);

min_n_power = 8; % for n = 256
max_n_power = 12; % for n = 4096
n_power_size = max_n_power - min_n_power + 1;

% Plot all of the information for each penalty parameter
n_powers = (min_n_power:max_n_power)';
n_values = 2.^n_powers;

% Truncate all elements below this threshold
threshold = 0.1;

repeats = 10;

t_fig = figure();
hold on;
dense_fig = figure();
hold on;
diff_fig = figure();
hold on;

penalty_functions = ["arctan", "cauchy", "L1", "L1-L2", "L1-double L2", "L1-half L2", "MCP", "SCAD", "TL1"];
for penalty_function_no = 1:length(penalty_functions)
    penalty_function_name = penalty_functions(penalty_function_no);
    
    dg = get_convex_derivative(penalty_function_name, lambda, a, theta_MCP, theta_SCAD, M_arctan);
    argmin_fn = get_argmin_fn_for_penalty(penalty_function_name, lambda, threshold_iterations, a, alpha_arctan, beta_arctan, gamma_arctan, gamma_cauchy);
    
    fprintf('Starting on penalty function %s\n', penalty_function_name);
    t = zeros(1, length(n_values));
    diff = zeros(1, length(n_values));
    dense = zeros(1, length(n_values));
    
    for i = 1:n_power_size
        n_power = n_powers(i);
        n = 2^n_power;
        fprintf('Starting %s with n = %d with %d repeats\n', penalty_function_name, n, repeats);

        % Reset rng so that we don't end up generating different matrices
        % for different penalty functions
        rng(rng_seed);

        % Generate a highly coherent matrix using the oversampled discrete cosine
        % transform from page 27 of https://arxiv.org/pdf/2003.04124.pdf
        P = m;
        F = 10;
        w = rand(1, P)';
        dct = @(w, j) 1/sqrt(P) .* cos(2.*pi.*w.*j./F);
        A_base = zeros(n, m);
        for j = 1:n
            A_base(j, :) = dct(w, j);
        end
        A_noise = matrix_noise*rand(n, m);
        A = A_base + A_noise;

        x_hat = zeros(m, repeats);
        for repeat = 1:repeats
            x_hat(:, repeat) = sprand(m, 1, density);
        end
        % Normalise x_hat to have maximum magnitude of 1
        %if (norm(x_hat, Inf) > 1)
        %    x_hat = x_hat ./ norm(x_hat, Inf);
        %end
        b_hat = A*x_hat;

        noise = normrnd(noise_mu, noise_sigma, n, repeats);
        b = b_hat + noise;

        x0 = A \ b;

        tic
        fprintf('Calculating solution to %s problem with n=%d\n', penalty_function_name, n);
        for repeat=1:repeats
            b_curr = b(:, repeat);
            x_hat_curr = x_hat(:, repeat);
            x0_curr = x0(:, repeat);
            [f, df, L] = get_objective_function('1D-L2', A, b_curr);
            
            obj_fn = @(x) (f(x) + penalty_1D_L1_L2(x, lambda, 1));
            stop_fn = @(x_prev, x_curr, iteration)(stop_fn_base(obj_fn, rtol, x0_curr, x_prev, x_curr, iteration));

            x_approx = ExtendedProximalDCMethod(f, df, L, x0_curr, dg, argmin_fn, stop_fn);
            
            dense(i) = dense(i) + nnz(truncate(x_approx, threshold));
            diff(i) = diff(i) + norm(x_approx - x_hat_curr, 2);
        end
        t(i) = toc;

        % divide by repeats to get average
        t(i) = t(i) / repeats;
        dense(i) = dense(i) / repeats;
        diff(i) = diff(i) / repeats;
    end
    
            
    figure(t_fig);
    plot(n_values, t, 'DisplayName', penalty_function_name);

    figure(dense_fig);
    plot(n_values, dense, 'DisplayName', penalty_function_name);

    figure(diff_fig);
    plot(n_values, diff, 'DisplayName', penalty_function_name);
end

figure(t_fig);
title('Average time taken (s)');
legend('Location', 'NorthWest');
ylabel('Time Taken (s)');
xlabel('Rows of A');

figure(dense_fig);
title('Average density');
legend('Location', 'NorthEast');
ylabel('Average density (# entires > 0.1)');
xlabel('Rows of A');

figure(diff_fig);
title('Average diff');
legend('Location', 'NorthEast');
ylabel('Average diff (2 norm of x - x_hat)');

function [argmin_fn] = get_argmin_fn_for_penalty(penalty_function_name, lambda, threshold_iterations, a, alpha_arctan, beta_arctan, gamma_arctan, gamma_cauchy)

argmin_fn_soft_lambda = get_argmin_function(lambda, 'L1', 'L2', threshold_iterations);
argmin_fn_soft_TL1 = get_argmin_function((a+1)/a, 'L1', 'L2', threshold_iterations);
argmin_fn_cauchy_lambda = get_argmin_function(lambda, 'cauchy', 'L2', threshold_iterations, 0, 0, gamma_cauchy);
argmin_fn_arctan_lambda = get_argmin_function(lambda, 'arctan', 'L2', threshold_iterations, alpha_arctan, beta_arctan, gamma_arctan);

switch penalty_function_name
    case 'arctan'
        argmin_fn = argmin_fn_arctan_lambda;
    case 'cauchy'
        argmin_fn = argmin_fn_cauchy_lambda;
    case 'TL1'
        argmin_fn = argmin_fn_soft_TL1;
    otherwise
        argmin_fn = argmin_fn_soft_lambda;
end
end