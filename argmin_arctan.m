function [x] = argmin_arctan(A, b, dD, w, xi, L, t, x_prev, lambda, max_eigval, thresholding_iterations, alpha, beta, gamma)
% Finds argmin {lambda*g(x) + <df(w) - xi, x-w> + L/2*||x-w||_2^2 + 1/t*D(x,
% x_prev)
% Where g(x) is the arctan penalty g(x) = sum(arctan(1+alpha*|t|)/beta) - arctan(1/beta)) + M*||x||_2^2 (the second term ensuring that the equation is
% convex with M > 2(alpha^2*beta/gamma)*(3/8)*(sqrt(1/beta^6))
% Taking M = 3*alpha^2*beta^(2/3)/(4*gamma)

% Differentiating the whole thing gives us the equation
% (alpha^2*(M+L)*x^3 +
% (alpha^2*(df-xi-Lw) + 2*alpha*(M+L)*sign(x))x^2 +
% (2*alpha(df-xi-Lw)*sign(x) + (beta^2+1)*(M+L)*sign(x))x +
% (alpha*beta/gamma * sign(x) + (beta^2+1)*(df-xi-Lw)) +
% 1/t* d/dx(D)
% = 0 as the stationary point

% TODO: For now we're assuming D(x, x_prev) = 1/2*||x||_2^2, as with other distance formulas we don't necessarily get a polynomial, 
% will need to make a case for each distance function
df = A'*(A*w - b);

M = (3*alpha^2*beta^(2/3))/(4*gamma);

n = length(x_prev);

x = x_prev;

% Two possible polynomials, when x is positive and when it is negative
% (based on sign)
% Since the problem is convex exactly one of them should have a zero within
% the domain


% Formulate it as ax^3 + bx^2 + cx + d
% Start with everything but the parts relying on D
a = alpha.^2.*(M+L).*ones(size(x_prev));
a_pos = a;
a_neg = a;

b_pos = ones(size(x_prev)).*(alpha.^2.*(df-xi-L.*w) + 2.*alpha.*(lambda.*M+L));
b_neg = ones(size(x_prev)).*(alpha.^2.*(df-xi-L.*w) - 2.*alpha.*(lambda.*M+L));

c_pos = ones(size(x_prev)).*(2.*alpha.*(df-xi-L.*w) + (beta^2+1).*(lambda.*M+L));
c_neg = -c_pos;

d_pos = ones(size(x_prev)).*(lambda.*alpha.*beta./gamma + (beta.^2+1).*(df-xi-L.*w));
d_neg = ones(size(x_prev)).*(-lambda.*alpha.*beta./gamma + (beta.^2+1).*(df-xi-L.*w));

% Add the terms based on D
% TODO: Currently assume using 1/2*||x-x_prev||_2^2
% For which d/dx = (x-x_prev)
a_pos = a_pos + alpha.^2./t;
a_neg = a_neg + alpha.^2./t;

b_pos = b_pos + (2.*alpha-alpha.^2.*x_prev)./t;
b_neg = b_neg + (-2.*alpha-alpha.^2.*x_prev)./t;

c_pos = c_pos + (beta.^2+1 - 2.*alpha.*x_prev)./t;
c_neg = c_neg + (beta.^2+1 + 2.*alpha.*x_prev)./t;

d_pos = d_pos - (beta.^2+1)./t;
d_neg = d_neg - (beta.^2+1)./t;

for i=1:n
    pos = slow_solve_cubic(a_pos(i), b_pos(i), c_pos(i), d_pos(i), @(x) x >= 0);
    neg = slow_solve_cubic(a_neg(i), b_neg(i), c_neg(i), d_neg(i), @(x) x <= 0);
    
    % Exactly one of the functions should have a solution within the
    % correct domain
    if (pos > 0) && (neg < 0)
        % Both functions have zero in the correct domain
        fprintf('Warning, got two zeros in correct domain %f %f\n', pos, neg);
        x(i) = pos;
    elseif (pos < 0) && (neg > 0)
        % Both functions have zero in wrong domain
        fprintf('Warning, got two zeros in incorrect domain %f %f\n', pos, neg);
        x(i) = pos;
    elseif (pos > 0) % neg >= 0
        x(i) = pos;
    elseif (neg < 0) % pos <= 0
        x(i) = neg;
    else % pos = 0, neg = 0
        x(i) = 0;
    end
end

end

