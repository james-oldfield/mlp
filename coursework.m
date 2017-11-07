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

% store previous gradients for momentum
d_weights_old = containers.Map(w_keys, zeros(1, 9));

% define hyperparams
eta = 0.25;
beta = 0.00;
n_epochs = 100;

errors = zeros(1, n_epochs);

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
        [h1, h2, h3, out] = forward(x_example, weights);
        
        % update the error for this batch (single example)
        i_error = i_error + abs(out - y_example);

        % -------------
        % BACKWARD PROP
        % -------------
        % perform backprop incrementally, after each example:
        d_weights = backward(h1, h2, h3, out, y_example, x1, x2, weights);

        % --------------
        % UPDATE WEIGHTS
        % --------------
        % update all weights in the map
        for k = keys(weights)
            weights(k{1}) = weights(k{1}) - eta * d_weights(k{1}) + ...
                beta * d_weights_old(k{1}); % momentum term
        end
                
        % update the old weights with the new
        d_weights_old = d_weights;

    end
    
    % store average error for this epoch
    fprintf('Error in epoch #%i', i_epoch);
    average_error = i_error / length(X);
    errors(i_epoch) = average_error;
    
    disp(average_error);
end

% plot the error decrease as a function of # epochs
plot(1:n_epochs, errors);
title('Error')
xlabel('Epoch #')
ylabel('Average Error across epoch')

