clear all;

% define our inputs and ground truth values:
X = [1 1
     1 0];
y = [0 1];

% define the initial weights provided by coursework,
% store in a Map:
w_keys = {'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9'};
w_vals = [-0.2, 0.15, -0.15, 0.1, 0.3, 0.1 ,0.3, 0.1, -0.3];
weights = containers.Map(w_keys, w_vals);

% define structure to store error derivatives
d_weights = containers.Map(w_keys, zeros(1, 9));

% define hyperparams, activ functions
eta = 0.1;

% define activations and their derivatives
sigmoid =       @(z) 1. / (1. + exp(-z));
sigmoid_prime = @(z) sigmoid(z) * (1 - sigmoid(z));

% ------------
% FORWARD PROP
% ------------
x_example = X(1, :);
y_example = y(1);

x1 = x_example(1);
x2 = x_example(2);

% calculate hidden units
h1 = sigmoid(weights('w1') * x1);
h3 = sigmoid(weights('w6') * x2);
h2 = sigmoid(weights('w3') * x1 + weights('w4') * x2);

% output 
out = weights('w9') * h1 ...
    + weights('w8') * h2 ...
    + weights('w7') * h3 ...
    + weights('w2') * x1 + weights('w5') * x2;
  
% -------------
% BACKWARD PROP
% -------------

% calculate betas at units
% activation fn = identity... deriv of identity wrt x = 1.
b_out = (out - y_example);

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

% --------------
% UPDATE WEIGHTS
% --------------

% update all weights in the map
for k = keys(weights)
    weights(k{1}) = weights(k{1}) - eta * d_weights(k{1});
end