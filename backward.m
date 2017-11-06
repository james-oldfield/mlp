function d_weights = backward(h1, h2, h3, out, y, x1, x2, weights)
%BACKWARD Compute deriv. of cost MSE wrt every input weight.
% Returns map of gradients for each weight.

% define map structure to store error derivatives,
% initialise to zero
d_weights = containers.Map(keys(weights), zeros(1, 9));

% calculate betas at units
% activation fn = identity... deriv of identity wrt x = 1.
b_out = (out - y);

b_h1 = weights('w9') * b_out * h1 * (1.0 - h1);
b_h2 = weights('w8') * b_out * h2 * (1.0 - h2);
b_h3 = weights('w7') * b_out * h3 * (1.0 - h3);

% calculate the derivatives of cost fn w/r/t each weight
d_weights('w9') = b_out * h1;
d_weights('w8') = b_out * h2;
d_weights('w7') = b_out * h3;

d_weights('w6') = b_h3 * x2;
d_weights('w4') = b_h2 * x2;
d_weights('w3') = b_h2 * x1;
d_weights('w1') = b_h1 * x1;

% linear terms
d_weights('w2') = b_out * x1;
d_weights('w5') = b_out * x2;

end