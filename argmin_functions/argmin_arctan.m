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

M = (2*alpha^2*beta)/(gamma*(1+beta^2));

n = length(x_prev);

x = x_prev;

% Two possible polynomials, when x is positive and when it is negative
% (based on sign)
% Since the problem is convex exactly one of them should have a zero within
% the domain


% Formulate it as ax^3 + bx^2 + cx + d
% Start with everything but the parts relying on D
a = (M*lambda*alpha^2 + L*alpha^2).*ones(size(x_prev));
a_pos = a;
a_neg = a;

b_pos = ones(size(x_prev)).*(2*alpha*M*lambda + alpha^2*df - xi*alpha^2 + 2*L*alpha - L*alpha^2*w);
b_neg = ones(size(x_prev)).*(-2*alpha*M*lambda + alpha^2*df - xi*alpha^2 - 2*L*alpha - L*alpha^2*w);

c_pos = ones(size(x_prev)).*(beta^2*M*lambda + M*lambda + 2*alpha*df - 2*xi*alpha + L*beta^2 + L - 2*L*alpha*w);
c_neg = ones(size(x_prev)).*(beta^2*M*lambda + M*lambda - 2*alpha*df + 2*xi*alpha + L*beta^2 + L + 2*L*alpha*w);

d_pos = ones(size(x_prev)).*(beta^2*df + df - beta^2*xi - xi - L*beta^2*w - L*w + alpha*beta*lambda/gamma);
d_neg = ones(size(x_prev)).*(beta^2*df + df - beta^2*xi - xi - L*beta^2*w - L*w - alpha*beta*lambda/gamma);

% Add the terms based on D
% TODO: Currently assume using 1/2*||x-x_prev||_2^2
% For which d/dx = (x-x_prev)
a_pos = a_pos + alpha^2./t;
a_neg = a_neg + alpha^2./t;

b_pos = b_pos + (2*alpha - alpha^2*x_prev)./t;
b_neg = b_neg + (-2*alpha - alpha^2*x_prev)./t;

c_pos = c_pos + (beta^2 + 1 - 2*alpha*x_prev)./t;
c_neg = c_neg + (beta^2 + 1 + 2*alpha*x_prev)./t;

d_pos = d_pos + (-beta^2*x_prev - x_prev)./t;
d_neg = d_neg + (-beta^2*x_prev - x_prev)./t;

for i=1:n
    pos = trig_solve_cubic(a_pos(i), b_pos(i), c_pos(i), d_pos(i), @(x) (x >= 0));
    neg = trig_solve_cubic(a_neg(i), b_neg(i), c_neg(i), d_neg(i), @(x) (x < 0));
    
    % Can have zeros at any of these points
    possible_minima = [pos, neg];
    
    % Since we've chosen an M large enough that g(x) + Mx^2 is convex,
    % there should be at most one stationary point which will be a global
    % minimum
    
    % And if there are no stationary points then that means that x(i) = 0
    % must be the global minimum (as the derivative for x(i) > 0 must be
    % negative and the derivative for x(i) < 0 must be positive)
    if length(possible_minima) == 1
        x(i) = possible_minima(1);
    elseif length(possible_minima) > 1
        fprintf('Warning: Got multiple results for solving the cauchy cubic\n');
        possible_minima
        
        min_value = Inf;
        min_point = 0;
        for j=1:length(possible_minima)
            curr_point = possible_minima(j);
            curr_value = penalty_arctan(curr_point, lambda, alpha, beta, gamma) + lambda*M/2*curr_point^2 + (df(i)-xi(i))*(curr_point-w(i)) + L/2*(curr_point - w(i))^2 + 1/(2*t)*(curr_point-x_prev(i))^2;

            %curr_value
            if curr_value < min_value
                min_value = curr_value;
                min_point = curr_point;
            end
        end
        %possible_minima
        %min_point
        %x_prev(i)

        %f_pos = @(x) (a_pos(i)*x.^3 + b_pos(i)*x.^2 + c_pos(i)*x + d_pos(i));
        %f_neg = @(x) (a_neg(i)*x.^3 + b_neg(i)*x.^2 + c_neg(i)*x + d_neg(i));


        %pos_vals = f_pos(possible_minima)
        %neg_vals = f_neg(possible_minima)

        %if (min_point == x_prev(i))
        %    g = @(x) (lambda*alpha*beta*sign(x)/(gamma) + ((alpha^2*x^2 + 2*alpha*abs(x) + beta^2 + 1))*(M*lambda*x + df(i) - xi(i) + L*(x-w(i)) + 1/t * (x-x_prev(i))));
        %    y = fzero(g, 0)
        %    g(y)
        %    f_pos(y)
        %    f_neg(y)
        %    diff = @(x)((g(x) - g(0)) - (f_pos(x) - f_pos(0)));
        %    error('dsgfdgs');
        %end

        x(i) = min_point;
    else
        % There is no stationary point, so 0 must be the minimum
        x(i) = 0;
    end 
end

end

