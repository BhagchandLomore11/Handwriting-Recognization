function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1); % You need to return the following variables correctly 

X = [ones(m, 1) X]; % Add ones to the X data matrix

z_two = X * Theta1';

a_two = sigmoid(z_two);

a_two = [ones(m, 1) a_two];

z_three = a_two * Theta2';

a_three = sigmoid(z_three);

[p_max, i_max] = max(a_three, [], 2);

p = i_max;

end
