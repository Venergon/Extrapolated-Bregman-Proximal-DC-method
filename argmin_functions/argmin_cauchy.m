function [x] = argmin_cauchy(df, dD, w, xi, L, t, x_prev, lambda, thresholding_iterations, gamma)
% Finds argmin {lambda*g(x) + <df(w) - xi, x-w> + L/2*||x-w||_2^2 + 1/t*D(x,
% x_prev)
% Where g(x) is the modified cauchy priory g(x) = -sum(log(gamma/(x_i^2 +
% gamma)) + M*||x||_2^2 (the second term ensuring that the equation is
% convex with M = 2

% Differentiating the whole thing gives us the equation
% (2*M + L)x_i^3 + (df(w) - xi - L*w_i)x_i^2 + (2 + 2*M*gamma + gamma*L)x_i
% + (gamma*df(w) - xi*gamma - gamma*L*w_i) + (x_i^2 + gamma)/2 * d/dx_i(D(x, x_prev))= 0 as the stationary point

% TODO: For now we're assuming D(x, x_prev) = 1/2*||x||_2^2, as with other bregman divergences we don't necessarily get a polynomial
df_w = df(w);

M = 2;

x = x_prev;

% Formulate it as ax^3 + bx^2 + cx + d
% Start with everything but the parts relying on D
a = (lambda*M + L)*ones(size(x));
b = df_w - xi - L*w;
c = (2*lambda*gamma + lambda*M*gamma + gamma*L)*ones(size(x));
d = gamma*df_w - xi*gamma - gamma*L*w;

% Add the terms based on D
% Those being x^2/(2t) *(d/dx D) + gamma/(2t)*(d/dx D)
% TODO: Currently assume using 1/2*||x-x_prev||_2^2
% For which d/dx = (x-x_prev)
a = a + 1/(t);
b = b - x_prev/(t);
c = c + gamma/(t);
d = d - gamma*x_prev/(t);

for i=1:length(x)
    sols = trig_solve_cubic(a(i), b(i), c(i), d(i), @(x) 1);
    if length(sols) ~= 1
        printf('Warning: have %d solutions, should only have 1', length(sols));
    end
    x(i) = sols(1);
end
end

