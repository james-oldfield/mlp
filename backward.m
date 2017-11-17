function [d_weights, d_linear] = backward(activations, y, x, weights, a_functions, linear_terms)
%BACKWARD Compute deriv. of cost MSE wrt every input weight.
% Returns array of gradients for each weight.
% ----
% :param activations: map of vectors of activations at each layer
% :param y: ground truth float
% :param x: input feature vector
% :param weights: map of matrices for weights for each layer
% :param a_functions: cell of activation function handles
% :param linear_terms: boolean (0,1) specifying whether to add lin terms
% ----
% Returns activations

d_weights = containers.Map;
b_errors  = containers.Map;

% add training examples into activations, so we can vectorise
activations('0') = x;

% loop over every layer in reverse, computing derivataives w/r/t weight
% matrices
for i=length(activations)-1 : -1:1
    this_layer = int2str(i);
    next_layer = int2str(i+1);
    prev_layer = int2str(i-1);
    
    % cache activation value
    % + store handle to derivative of activ fn. used at this layer
    a = activations(this_layer);
    a_fn_d = @(x) a_functions{i}(x, 1);

    % ---------------
    % ERRORS AT UNITS
    % ---------------
    % compute dE/dZ for output layer, in vectorised form,
    % equations from http://neuralnetworksanddeeplearning.com/chap2.html
    if i == length(activations)-1
        % take hadamard product of error * deriv. of activ function
        % as per equation BP1 ? ?^L = (a^L?y) ? ??(z^L).
        b_error = (a - y) .* arrayfun(a_fn_d, a);
        
        % compute deriv w/r/t linear weights
        d_linear = b_error * x;
    else
        % otherwise we backpropagate the error
 
        % per equation BP2 ? ?l=((w^l+1)T ?^l+1) ? ??(z^l),
        b_error = weights(next_layer)' * b_errors(next_layer) .* arrayfun(a_fn_d, a);
    end
    
    % store the vector of derivatives for units at this layer
    b_errors(this_layer) = b_error;
    
    % -----------------
    % ERRORS AT WEIGHTS
    % -----------------
    % backpropogate again to find deriv. w/r/t weights themselves
    d_weights(this_layer) = b_errors(this_layer)' * activations(prev_layer);
end