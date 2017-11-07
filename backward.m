function d_weights = backward(activations, y, x1, x2, weights)
%BACKWARD Compute deriv. of cost MSE wrt every input weight.
% Returns array of gradients for each weight.

% define structure to store error derivatives,
% initialise to zero
d_weights = zeros(1, 9);

% calculate betas at units
% activation fn = identity... deriv of identity wrt x = 1.
b_out = (activations('out') - y);

b_h1 = weights(9) * b_out * activations('h1') * (1.0 - activations('h1'));
b_h2 = weights(8) * b_out * activations('h2') * (1.0 - activations('h2'));
b_h3 = weights(7) * b_out * activations('h3') * (1.0 - activations('h3'));

% calculate the derivatives of cost fn w/r/t each weight
d_weights(9) = b_out * activations('h1');
d_weights(8) = b_out * activations('h2');
d_weights(7) = b_out * activations('h3');

d_weights(6) = b_h3 * x2;
d_weights(4) = b_h2 * x2;
d_weights(3) = b_h2 * x1;
d_weights(1) = b_h1 * x1;

% linear terms
d_weights(2) = b_out * x1;
d_weights(5) = b_out * x2;

end