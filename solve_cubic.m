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
sols = roots([a, b, c, d]);
real_roots = sols(imag(sols) == 0);
matching_roots = real_roots(cond(real_roots));

if length(matching_roots) > 1
    fprintf('Warning: %d real roots\n', length(matching_roots));
end

if length(matching_roots) == 0
    x = 0;
else
    x = matching_roots(1);
end
end

