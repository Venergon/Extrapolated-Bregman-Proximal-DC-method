function [dO] = objective_derivative_2D_image_filter(X, Bobs, P)
    dO = 2*imfilter(imfilter(X,P,'symmetric')-Bobs,P,'symmetric');
end
