function [x_truncated] = truncate(x, threshold)
%TRUNCATE_SMALL_ELEMENTS Given a list of elements, sets all elements with a
%an absolute value less than the threshold to 0

% Inputs:
%  x: the vector to truncate
%  tolerance: The value for which every element smaller than it will be set
%  to 0
x_truncated = x;

for i=1:length(x_truncated)
    if abs(x_truncated(i)) < threshold
        x_truncated(i) = 0;
    end
end
end