function a = identity(a, derivative)
%IDENTITY (i.e. identity activation function) for summation layer
% ----
% :param a: input
% :param derivative: boolean?return derivative?
% ----
% Returns identity or derivative
if derivative == 1
    % return derivative, for backprop.
    a = 1;
end
end