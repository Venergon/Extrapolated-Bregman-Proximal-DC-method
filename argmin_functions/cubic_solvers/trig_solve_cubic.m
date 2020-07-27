function [x] = trig_solve_cubic(a, b, c, d, cond)
% Given a cubic equation of the form ax^3 + bx^2 + cx + d, get a real root
% of that equation, matching the given condition
% Gives a warning if there is more than one real root matching the
% condition

% Inputs:
%  a: coefficient of x^3
%  b: coefficient of x^2
%  c: coefficient of x
%  d: constant
%  cond: Condition for the equation to match

% Output:
%  x: one of the real roots of the equation

% Uses the 'Trigonometric solution for three real roots' from https://en.wikipedia.org/wiki/Cubic_equation

% First transform the cubic from ax^3 + bx^2 + cx + d
% into the depressed cubic t^3 + pt + q,
% where x = t - b/(3a)
p = (3*a*c - b^2)/(3*a^2);
q = (2*b^3 - 9*a*b*c + 27*a^2*d)/(27*a^3);

if (p == 0)
    t = nthroot(q, 3);
elseif (4*p^3 + 27*q^2 < 0)
    % Multiple real roots
    root_formula = @(k) (2*sqrt(-p/3)*cos(1/3*acos((3*q)/(2*p) * sqrt(-3/p) - 2*pi*k/3)));
    t = [root_formula(0), root_formula(1), root_formula(2)];
else
    % One real root
    if (p < 0)
        t = -2*abs(q)/q*sqrt(-p/3)*cosh(1/3*acosh((-3*abs(q))/(2*p)*sqrt(-3/p)));
    else
        t = -2*sqrt(p/3)*sinh(1/3*asinh((3*q)/(2*p)*sqrt(3/p)));
    end
end

x = t - b/(3*a);

real_roots = x(imag(x) == 0);
matching_roots = real_roots(cond(real_roots));

x = matching_roots;
end