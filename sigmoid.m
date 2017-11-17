function activation = sigmoid(z, derivative)
%SIGMOID Compute sigmoid activation of z, or its derivative
% ----
% :param z: weighted summation, pre-activation
% :param derivative: boolean?return derivative?
% ----
% Returns sigmoid(z) or sigmoid'(z)
activation = 1. / (1. + exp(-z));

if derivative == 1
    activation = z * (1-z);
end

