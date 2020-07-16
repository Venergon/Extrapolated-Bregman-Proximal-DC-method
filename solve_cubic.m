function [x] = solve_cubic(a, b, c, d, cond)
% Given a cubic equation of the form ax^3 + bx^2 + cx + d, get a real root
% of that equation, matching a condition

% Inputs:
%  a: coefficient of x^3
%  b: coefficient of x^2
%  c: coefficient of x
%  d: constant
%  cond: Condition for the equation to match

% Output:
%  x: one of the real roots of the equation

% TODO: roots might possibly be slow if it doesn't use a specific algorithm
% for cubics, check that and see if we need to use the cubic formula
% instead

% Uses the 'General Cubic Formula' from https://en.wikipedia.org/wiki/Cubic_equation
delta_0 = b^2 - 3*a*c;
delta_1 = 2*b^3 - 9*a*b*c + 27*a^2*d;

C = nthroot(((delta_1 + sqrt(delta_1^2 - 4*delta_0^3))/2), 3);
if (C == 0)
    C = nthroot(((delta_1 - sqrt(delta_1^2 - 4*delta_0^3))/2), 3);
end

xi = (-1 + 1j*sqrt(3))/2;

solution_fn = @(k) (-1/(3*a) * (b + xi^k*C + delta_0/(xi^k*C)));
sols = [solution_fn(0), solution_fn(1), solution_fn(2)];

real_roots = sols(imag(sols) == 0);
matching_roots = real_roots(cond(real_roots));

if length(matching_roots) > 1
    fprintf('Warning: %d real roots\n', length(matching_roots));
end

if length(matching_roots) == 0
    x = real_roots(1);
else
    x = matching_roots(1);
end
end

