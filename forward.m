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
    
    % anon function, activation at this layer (no derivative - 0)
    a_fn = @(x) a_functions{i}(x, 0);
    
    % vectorised computation of activations, and store in map
    % using the specified activation function handle (e.g. sigmoid)
    a = arrayfun(a_fn, a * weights(layer_i));
    
    % add linear terms in last iteration
    % (inputs x linear_weights)
    if i == length(weights)
        a = a + x * linear_terms;
    end
    
    fprintf("Activations at layer %d", i);
    disp(a);
    
    activations(layer_i) = a;
end