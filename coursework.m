clear all;

% define our inputs and ground truth values:
X = [1 1
     1 0];
y = [0 1];

% define the initial weights provided by coursework,
weights = [-0.2 0.15 -0.15 0.1 0.3 0.1 0.3 0.1 -0.3];

% store previous gradients for momentum
d_weights_old = zeros(1, 9);

% define hyperparams
eta = 0.75;
beta = 0.2;
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
        activations = forward(x_example, weights);
        
        % update the error for this batch (single example)
        i_error = i_error + abs(activations('out') - y_example);

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

