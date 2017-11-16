function activations = forward(x, weights, a_functions, linear_terms)
%FORWARD forward propagates network
% ----
% :param x: input vector for mini-batch
% :param weights: map of matrices for weights for each layer
% :param a_functions: cell of activation function handles
% :param linear_terms: boolean (0,1) specifying whether to add lin terms
% ----
% Returns activations

% create a map of matrices to store activations at each layer
activations = containers.Map;

% first activation is simply our inputs
a = x;

% propagate signal through each layer
for i = 1:length(weights)
    layer_i = int2str(i);
    a_fn = a_functions{i};
    
    % vectorised computation of activations, and store in map
    % using the specified activation function handle (e.g. sigmoid)
    a = arrayfun(a_fn, a * weights(layer_i));
    activations(layer_i) = a;
end

% get the last layer's output
last_layer = int2str(length(activations));
out = activations(last_layer);

% add on linear terms to output if specified as 1, else zeros out.
activations(last_layer) = out + x * linear_terms;