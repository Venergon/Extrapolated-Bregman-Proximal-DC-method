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
a = (lambda*M + L).*ones(size(x_prev));
a_pos = a;
a_neg = a;

b_pos = ones(size(x_prev)).*(6*lambda*M + df - xi + 6*alpha*L - w*L);
b_neg = ones(size(x_prev)).*(-6*lambda*M + df - xi - 6*alpha*L - w*L);

c_pos = ones(size(x_prev)).*(4*lambda*M + 6*alpha*df - 6*alpha*xi + 4*L - 6*alpha*w*L);
c_neg = ones(size(x_prev)).*(4*lambda*M - 6*alpha*df + 6*alpha*xi + 4*L + 6*alpha*w*L);

d_pos = ones(size(x_prev)).*(6*sqrt(3)*lambda*alpha/pi + 4*df - 4*xi - 4*L*w);
d_neg = ones(size(x_prev)).*(-6*sqrt(3)*lambda*alpha/pi + 4*df - 4*xi - 4*L*w);

% Add the terms based on D
% TODO: Currently assume using 1/2*||x-x_prev||_2^2
% For which d/dx = (x-x_prev)
a_pos = a_pos + 1./t;
a_neg = a_neg + 1./t;

b_pos = b_pos + (6.*alpha-x_prev)./t;
b_neg = b_neg + (-6.*alpha-x_prev)./t;

c_pos = c_pos + (4-6.*alpha.*x_prev)./t;
c_neg = c_neg + (4+6.*alpha.*x_prev)./t;

d_pos = d_pos + (-4.*x_prev)./t;
d_neg = d_neg + (-4.*x_prev)./t;

for i=1:n
    pos = trig_solve_cubic(a_pos(i), b_pos(i), c_pos(i), d_pos(i), @(x) 1);
    neg = trig_solve_cubic(a_neg(i), b_neg(i), c_neg(i), d_neg(i), @(x) 1);
    
    % Can have zeros at any of these points
    possible_minima = [pos, neg, 0];
    
    min_value = Inf;
    min_point = 0;
    for j=1:length(possible_minima)
        curr_point = possible_minima(j);
        curr_value = penalty_arctan(curr_point, lambda, alpha, beta, gamma) + lambda*M/2*curr_point^2 + (df(i)-xi(i))*(curr_point-w(i)) + L/2*(curr_point - w(i))^2 + 1/(2*t)*(curr_point-x_prev(i))^2;

        if curr_value < min_value
            min_value = curr_value;
            min_point = curr_point;
        end
    end
    %possible_minima
    %min_point
    %x_prev(i)

    x(i) = min_point;
end

end

