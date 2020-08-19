function [O] = objective_2D_image_filter(X, Bobs, P)
    O = norm(imfilter(X,P,'symmetric') - Bobs,'fro')^2;
end
