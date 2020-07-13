function [x] = argmin_cauchy(A, b, dD, w, xi, L, t, x_prev, lambda, max_eigval, thresholding_iterations, gamma)
% Finds argmin {lambda*g(x) + <df(w) - xi, x-w> + L/2*||x-w||_2^2 + 1/t*D(x,
% x_prev)
% Where g(x) is the modified cauchy priority g(x) = -sum(log(gamma/(x_i^2 +
% gamma)) + M*||x||_2^2 (the second term ensuring that the equation is
% convex with M = 2

% Differentiating the whole thing gives us the equation
% (2*M + L)x_i^3 + (df(w) - xi - L*w_i)x_i^2 + (2 + 2*M*gamma + gamma*L)x_i
% + (gamma*df(w) - xi*gamma - gamma*L*w_i) + (x_i^2 + gamma)/2 * d/dx_i(D(x, x_prev))= 0 as the stationary point

% TODO: For now we're assuming D(x, x_prev) = 0, as with that the whole
% equation is a cubic which can be solved, will need to make a case for
% each distance function
df = A'*(A*w - b);

M = 2;

n = length(x_prev);

x = zeros(size(x_prev));

% Formulate it as ax^3 + bx^2 + cx + d

% a, c do not depend on index
a = (2*lambda*M + L)*ones(size(x));
c = (2*lambda + 2*lambda*M*gamma + gamma*L)*ones(size(x));

b = df - xi - L*w;
d = gamma*df - xi*gamma - gamma*L*w;

for i=1:n
    x(i) = solve_cubic(a(i), b(i), c(i), d(i));
end
%for iteration=1:thresholding_iterations
%    for i=1:n
%        b = df(i) - xi(i) - L*w(i);
%        d = gamma*df(i) - xi(i)*gamma - gamma*L*w(i);
%
%        x(i) = solve_cubic(a, b, c, d);
%    end
%end

end

