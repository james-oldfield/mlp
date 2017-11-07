function activations = forward(x, weights)
%FORWARD forward propagates network using supplied variables.
% Returns activations
a_keys = {'h1', 'h2', 'h3', 'out'};
activations = containers.Map(a_keys, zeros(1, 4));

x1 = x(1);
x2 = x(2);

% calculate hidden units
activations('h1') = sigmoid(weights(1) * x1);
activations('h3') = sigmoid(weights(6) * x2);
activations('h2') = sigmoid(weights(3) * x1 + weights(4) * x2);

% output (unthresholded neuron?i.e. identity activation)
activations('out') = weights(9) * activations('h1') ...
    + weights(8) * activations('h2') ...
    + weights(7) * activations('h3') ...
    + weights(2) * x1 + weights(5) * x2;
  
end

