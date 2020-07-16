function [x] = argmin_cauchy(A, b, dD, w, xi, L, t, x_prev, lambda, max_eigval, thresholding_iterations, gamma)
% Finds argmin {lambda*g(x) + <df(w) - xi, x-w> + L/2*||x-w||_2^2 + 1/t*D(x,
% x_prev)
% Where g(x) is the modified cauchy priory g(x) = -sum(log(gamma/(x_i^2 +
% gamma)) + M*||x||_2^2 (the second term ensuring that the equation is
% convex with M = 2

% Differentiating the whole thing gives us the equation
% (2*M + L)x_i^3 + (df(w) - xi - L*w_i)x_i^2 + (2 + 2*M*gamma + gamma*L)x_i
% + (gamma*df(w) - xi*gamma - gamma*L*w_i) + (x_i^2 + gamma)/2 * d/dx_i(D(x, x_prev))= 0 as the stationary point

% TODO: For now we're assuming D(x, x_prev) = 1/2*||x||_2^2, as with other distance formulas we don't necessarily get a polynomial, 
% will need to make a case for each distance function
df = A'*(A*w - b);

M = 2;

n = length(x_prev);

x = x_prev;

% Formulate it as ax^3 + bx^2 + cx + d
% Start with everything but the parts relying on D
a = (2*lambda*M + L)*ones(size(x));
b = df - xi - L*w;
c = (2*lambda + 2*lambda*M*gamma + gamma*L)*ones(size(x));
d = gamma*df - xi*gamma - gamma*L*w;

% Add the terms based on D
% Those being x^2/(2t) *(d/dx D) + gamma/(2t)*(d/dx D)
% TODO: Currently assume using 1/2*||x-x_prev||_2^2
% For which d/dx = (x-x_prev)
a = a + 1/(2*t);
b = b - x_prev/(2*t);
c = c + gamma/(2*t);
d = d - gamma*x_prev/(2*t);

for i=1:n
    x(i) = solve_cubic(a(i), b(i), c(i), d(i));
end

end

