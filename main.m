clear all;

% define our inputs and ground truth values:
X = [1 1
     1 0];
y = [0 1];

% boolean to specify whether we use coursework weights
coursework = 1;

% --------------------
% SPECIFY ARCHITECTURE
% --------------------

% i.e. [2 3 1] specifies that 2 units are in the input layer,
% 3 units in hidden layer
% 1 unit in the output
architecture = [2 3 1];

% specify weights for linear terms
% modify fn to use zeros() if not required
linear_terms = rand(size(X, 2)) - 0.5;

% specify which activ fn we wish to use at each layer,
% storing function handles.
a_functions  = {@sigmoid, @identity};

% create map of matrices to contain weights for every layer,
% and corresponding derivatives
weights   = containers.Map;
d_weights = containers.Map;

% populate weight mats with zeros to map from layer i to i+1
for i = 1:length(architecture)-1
    weights(int2str(i)) = rand(architecture(i), architecture(i+1)) - 0.5;
    d_weights(int2str(i)) = zeros(architecture(i), architecture(i+1));
end

% populate weights with coursework values, if chosen, else remain random.
if coursework == 1
    % load coursework weights
    load w1.dat;
    load w2.dat;
    load linear_weights.dat;

    weights('1') = w1;
    weights('2') = w2;
    linear_terms = linear_weights;
end

% define hyperparams
eta = 0.75;
beta = 0.2;
n_epochs = 100;

last_layer = int2str(length(architecture)-1);
errors = zeros(1, n_epochs);

% -----
% TRAIN
% -----

for i_epoch = 1:n_epochs
    i_error = 0;
    
    % loop through examples in training set
    for i_example = 1:length(X)
        x_example = X(i_example, :);
        y_example = y(i_example);

        % get features of this example
        x1 = x_example(1);
        x2 = x_example(2);

        % ------------
        % FORWARD PROP
        % ------------
        % calculate the activations at all nodes
        activations = forward(x_example, weights, a_functions, linear_terms);
        
        % update the error for this batch (single example)
        i_error = i_error + abs(activations(last_layer) - y_example);

        % -------------
        % BACKWARD PROP
        % -------------
        % perform backprop incrementally, after each example:
        d_weights = backward(activations, y_example, x1, x2, weights);

        % --------------
        % UPDATE WEIGHTS
        % --------------
        % update momentum term
        d_weights_old = beta * d_weights_old + d_weights;
        
        % update weight vector
        weights = weights - eta * d_weights_old;

    end
    
    % store average error for this epoch
    fprintf('Error in epoch #%i', i_epoch);
    average_error = i_error / length(X);
    errors(i_epoch) = average_error;
    
    disp(average_error);
end

% plot the error decrease as a function of # epochs
plot(1:n_epochs, errors);
title(sprintf('Error curves - eta: %.2f - beta: %.2f', eta, beta))
xlabel('Epoch #')
ylabel('Average Error across epoch')

