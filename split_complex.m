function [x_split] = split_complex(x_combined)
% Split a vector of complex numbers into two vectors and append the complex
% vector to the end of the real vector
x_real = real(x_combined);
x_imag = imag(x_combined);

x_split = [x_real; x_imag];
end

