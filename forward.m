function [h1, h2, h3, out] = forward(x, weights)
%FORWARD forward propagates network using supplied variables.
% Returns activations

x1 = x(1);
x2 = x(2);

% calculate hidden units
h1 = sigmoid(weights('w1') * x1);
h3 = sigmoid(weights('w6') * x2);
h2 = sigmoid(weights('w3') * x1 + weights('w4') * x2);

% output (unthresholded neuron?i.e. identity activation)
out = weights('w9') * h1 ...
    + weights('w8') * h2 ...
    + weights('w7') * h3 ...
    + weights('w2') * x1 + weights('w5') * x2;
  
end

