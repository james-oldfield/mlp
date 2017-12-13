% boolean to specify whether we use coursework weights,
% this also zeros-out connections for a non-fully connected net.
coursework = 1;
% use exponential learning rate decay?
exp_eta_decay = 0;

% --------------------
% SPECIFY ARCHITECTURE
% --------------------
% i.e. [2 3 5 1] specifies that 2 units are in the input layer,
% 3 units in hidden layer
% 5 units in next hidden layer, etc...
% 1 unit in the output
architecture = [2 3 1];

% specify which activ fn we wish to use at each layer,
% storing function handles.
% MUST specify an activ function for each layer.
a_functions  = {@sigmoid, @identity};

last_layer = int2str(length(architecture)-1);

% ----------------
% POPULATE WEIGHTS
% ----------------
% create map of matrices to contain weights for every layer,
% and corresponding derivatives
weights       = containers.Map;
d_weights_old = containers.Map;

% populate weight mats with rand(-0.5,0.5) to map from layer i to i+1
for i = 1:length(architecture)-1
    weights(int2str(i)) = rand(architecture(i), architecture(i+1)) - 0.5;
    d_weights_old(int2str(i)) = zeros(architecture(i), architecture(i+1));
end

% specify weights for linear terms
% modify fn to use zeros() if not required
linear_terms = rand(length(X), 1) - 0.5;
d_linear_old = zeros('like', linear_terms);

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

% load data
load X.dat;
load y.dat;

% ---------------
% HYPERPARAMETERS
% ---------------
eta = 0.1;
beta = 0.25;
n_epochs = 100;

% use coursework's hyperparams if desired
if coursework
    eta = 1.0;
    beta = 0.0; % no momentum!
    n_epochs = 1;
end

errors = zeros(1, n_epochs);

% -----
% TRAIN
% -----

for i_epoch = 1:n_epochs
    i_error = 0;
    
    % decay learning rate exponentially
    if exp_eta_decay
        eta = eta * exp(-0.01 * i_epoch);
    end
    
    % loop through examples in training set
    for i_example = 1:length(X)
        x_example = X(i_example, :);
        y_example = y(i_example);

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
        % perform backprop incrementally, after each example,
        % i.e. compute error derivatives
        [d_weights, d_linear] = backward(activations, y_example, x_example, weights, a_functions);
        
        % ---------
        % 'DROPOUT'
        % ---------
        % If we're using coursework, then nix any derivatives for
        % connections we're not using (i.e. weight == 0).
        % This prevents the network being fully connected.
        if coursework
            d_weights('1') = d_weights('1') .* (w1 ~= 0);
            d_weights('2') = d_weights('2') .* (w2 ~= 0);
        end
        
        % ----------
        % LOG WEIGHTS
        % ----------
        fprintf("Errors at weights, deltas, at layer %d", i);
        celldisp(values(d_weights));
        
        fprintf("Errors at linear weights:");
        disp(d_linear);

        % --------------
        % UPDATE WEIGHTS
        % --------------
        % update each weight matrix in the map using incremental G.D.
        for i=1:length(weights)
            this_layer = int2str(i);
            
            % update momentum term
            d_weights_old(this_layer) = beta * d_weights_old(this_layer) + d_weights(this_layer);
            % update weight matrix
            weights(this_layer) = weights(this_layer) - eta * d_weights_old(this_layer);
        end
        
        % udpate the linear terms vector
        d_linear_old = beta * d_linear_old + d_linear';
        linear_terms = linear_terms - eta * d_linear_old;
        
        fprintf("NEW WEIGHTS, epoch %d:", i_epoch);
        celldisp(values(weights));
        fprintf("NEW LINEAR TERMS WEIGHTS, epoch %d:", i_epoch);
        disp(linear_terms);
    end
    
    % store average error for this epoch
    fprintf('Error in epoch #%i', i_epoch);
    average_error = i_error / length(X);
    errors(i_epoch) = average_error;
    
    disp(average_error);
end

if ~coursework
    % plot the error decrease as a function of # epochs
    plot(1:n_epochs, errors);
    title(sprintf('Error curves - eta: %.2f - beta: %.2f', eta, beta))
    xlabel('Epoch #')
    ylabel('Average Error across epoch')
end