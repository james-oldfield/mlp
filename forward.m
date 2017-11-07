function [h1, h2, h3, out] = forward(x, weights)
%FORWARD forward propagates network using supplied variables.
% Returns activations

x1 = x(1);
x2 = x(2);

% calculate hidden units
h1 = sigmoid(weights(1) * x1);
h3 = sigmoid(weights(6) * x2);
h2 = sigmoid(weights(3) * x1 + weights(4) * x2);

% output (unthresholded neuron?i.e. identity activation)
out = weights(9) * h1 ...
    + weights(8) * h2 ...
    + weights(7) * h3 ...
    + weights(2) * x1 + weights(5) * x2;
  
end

