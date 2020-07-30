function [x_combined] = combine_complex(x_split)
% Recombine a vector that had been split by split_complex
x_real = x_split(1:end/2);
x_imag = x_split(end/2 + 1:end);

x_combined = x_real + 1i*x_imag;
end

