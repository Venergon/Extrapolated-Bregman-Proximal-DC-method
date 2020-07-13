function [x] = solve_cubic(a, b, c, d)
% Given a cubic equation of the form ax^3 + bx^2 + cx + d, get a real root
% of that equation

% Inputs:
%  a: coefficient of x^3
%  b: coefficient of x^2
%  c: coefficient of x
%  d: constant

% Output:
%  x: one of the real roots of the equation

% Will warn if there are multiple roots

% TODO: roots might possibly be slow if it doesn't use a specific algorithm
% for cubics, check that and see if we need to use the cubic formula
% instead
sols = roots([a, b, c, d]);
real_roots = sols(imag(sols) == 0);

if length(real_roots) ~= 1
    fprintf('Warning: %d real roots\n', length(real_roots));
end

x = real_roots(1);
end

