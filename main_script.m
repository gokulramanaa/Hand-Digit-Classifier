%%Hand digit classifier using Neural Network 
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that "0" mapped to label 10)

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')
load('number_data.mat');
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
seld = sel(1:100);
displayData(X(seld, :));
fprintf('Program paused. Press enter to continue.\n');
pause;

% Splitting 90% of the data for training and 10% data for testing
X_train = X(sel(1:4500),:);
y_train = y(sel(1:4500),:);
X_test = X(sel(4501:end),:);
y_test = y(sel(4501:end),:);


%% Initializing Pameters 
fprintf('\nInitializing Neural Network Parameters ...\n')
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
%
% Unroll parameters
nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% Implement Regularization

fprintf('\nChecking Backpropagation (w/ Regularization) ... \n')
%  Check gradients by running checkNNGradients
lambda = 3;
checkNNGradients(lambda);
% Also output the costFunction debugging values
debug_J  = nnCostFunction(nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X_train, y_train, lambda);

fprintf(['\n\nCost at (fixed) debugging parameters (w/ lambda = %f): %f ' ...
         '\n\n'], lambda, debug_J);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% Training NN 

fprintf('\nTraining Neural Network... \n')
options = optimset('MaxIter', 50);

%lambda can be tried with different values
lambda = 1;

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_train, y_train, lambda);


[nn_params, cost] = fmincg(costFunction, nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================= Implement Predict =================
%pred = predict(Theta1, Theta2, X_train);
pred1 = predict(Theta1,Theta2,X_test);

%fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y_train)) * 100);
fprintf('\Testing Set Accuracy: %f\n', mean(double(pred1 == y_test)) * 100);



