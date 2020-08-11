% Test Extended Bregman Proximal DC method using an image against the ISTA
% method on page 197 of https://web.iem.technion.ac.il/images/user-files/becka/papers/71654.pdf

X = double(imread('images/cameraman.pgm'));
X = X/255;
[P, center] = psfGauss([9, 9], 4);
B = imfilter(X, P, 'symmetric');

randn('seed', 314);
Bobs = B + 1e-3*randn(size(B));

subplot(1, 2, 1);
imshow(X, []);
subplot(1, 2, 2);
imshow(Bobs, []);