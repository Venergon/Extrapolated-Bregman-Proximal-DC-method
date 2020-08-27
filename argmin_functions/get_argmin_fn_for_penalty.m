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