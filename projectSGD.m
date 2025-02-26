
clear; % refresh workspace

%%  BEGIN DATA PROCESSING SECTION 


% Loading and Reading our data
dataframe_temp = readtable('Housing.csv'); 

% Declare the categorical variables in our raw data
categorical_vars = {'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus'};

% Change them to true false form
for var = categorical_vars
    dataframe_temp.(var{1}) = strcmp(dataframe_temp.(var{1}), 'yes');
end

% Declare our numerical variables in our raw data
% Then normalize them to help controlling the scale of the data
numeric_vars = removevars(dataframe_temp, categorical_vars); % excluding cat_variables
numeric_vars = normalize(numeric_vars, 'zscore');

% Convert categorical variables to take binary form 1 (for true) and 0 (for false)
logical_vars = dataframe_temp{:, categorical_vars};
logical_vars = array2table(double(logical_vars), 'VariableNames', categorical_vars);
logical_vars = normalize(logical_vars, 'zscore');

% The dataframe consisting of our normalized numerical variables and
% numeric values of the categorical variables
dataframe = [numeric_vars, logical_vars];

disp(head(dataframe));

% Specify the model parameters : predictors and features (price)
predictors = {'area', 'bedrooms', 'bathrooms', 'stories', 'parking','mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea'};
features = {'price', 'bedrooms', 'bathrooms', 'stories', 'parking','mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea'};



%% Setting up Training Data

% creating a matrix to store the features we want to include in our
% training data. These features show higher correlation in the heatmap than
% those that aren't included into the selected_features matrix variable
selected_features = dataframe{:, features};

rng(1); % For reproducibility. This is what makes SGD stochastic 

% Create a holdout validation set (80% training, 20% testing)
cv = cvpartition(size(selected_features, 1), 'HoldOut', 0.2);
idxTrain = training(cv);

% Training data set up
X_train = selected_features(idxTrain, 2:end);
Y_train = selected_features(idxTrain, 1);

% Testing data set up
X_test = selected_features(~idxTrain, 2:end);
Y_test = selected_features(~idxTrain, 1);

% Feature scaling for training set (for better convergence reason)
mu = mean(X_train);
sigma = std(X_train);
X_train_scaled = (X_train - mu) ./ sigma;

% Feature scaling for testing set
X_test_scaled = (X_test - mu) ./ sigma;

% Display sizes for verification
%disp(size(X_train)); 
%disp(size(Y_train));
%disp(size(X_test));
%disp(size(Y_test));
%disp(size(X_train_scaled));


%%  Stochastic Gradient Descent Implementation

% Set hyperparameters
alpha = 0.1;  % Learning rate
num_iterations = 100;  % Number of iterations
restol = 1e-8;

% Run SGD
fprintf('Stochastic Gradient Descent: \n');
[theta, cost_history] = SGD(X_train_scaled, Y_train, alpha, num_iterations, restol);

% Display the learned parameters
disp('Learned Parameters:');
disp(theta);

% Plot the cost history to monitor convergence
figure(4); clf;
plot(1:num_iterations, cost_history, '-o');
xlabel('Iteration');
ylabel('Mean Squared Error');
title('Convergence of Stochastic Gradient Descent');


%% Steepest Descent Implementation

% Split data into training and testing sets again
cv2 = cvpartition(size(selected_features, 1), 'HoldOut', 0.2);
idxTrain2 = training(cv2);
training_data2 = selected_features(idxTrain2, :);
testing_data2 = selected_features(~idxTrain2, :);

% Separate input features (X) and target variable (y) for training set
X_train2 = training_data2(:, 2:end);
y_train2 = training_data2(:, 1);

% COnfigurations for the steepest descent method
num_iterations2 = 100;
restol2 = 1e-8;
% alpha will be decided by the lineSearch method in the steepest descent
% implementation

% Call the steepest descent method function 
fprintf('Stochastic Gradient Descent: \n');
[theta2, cost_history2] = steepestDescentLineSearch(X_train_scaled, Y_train, num_iterations2, restol2);

% Display the learned parameters
disp('Learned Parameters:');
disp(theta2);

% Plot the cost history to monitor convergence
figure;
plot(1:length(cost_history2), cost_history2, '-ro');
xlabel('Iteration');
ylabel('Mean Squared Error');
title('Convergence of Steepest Descent');



%% EVALUATION AND ANALYSIS 

% Plot the cost history to monitor convergence for both methods altogether
figure(4); clf;
plot(1:num_iterations, cost_history, '-o', 'LineWidth',1);
xlabel('Iteration');
ylabel('Mean Squared Error');
hold on;
plot(1:length(cost_history2), cost_history2, '-o', 'LineWidth',1);
xlabel('Iteration');
ylabel('Mean Squared Error');
title('Convergence of SGD and Steepest Descent');
legend('SGD', 'Steepest Descent');
hold off;

%% Scatter plot to show Predicted Price and Actual Price 

Y_pred_SGD = X_test_scaled * theta;

% Scatter plot for SGD
figure;
scatter(Y_test, Y_pred_SGD, 'bo', 'DisplayName', 'Predicted Prices');
hold on;
plot(Y_test, Y_test, 'r-', 'LineWidth',3 , 'DisplayName', 'Actual Prices');
xlabel('Actual House Prices');
ylabel('Predicted House Prices');
title('Scatter Plot - Stochastic Gradient Descent');
legend('show');
grid on;

Y_pred_Steepest = X_test * theta2;

% Scatter plot for SD
figure;
scatter(Y_test, Y_pred_Steepest, 'b', 'DisplayName', 'Predicted Prices');
hold on;
plot(Y_test, Y_test, 'r-', 'LineWidth',3 ,'DisplayName', 'Actual Prices');
xlabel('Actual House Prices');
ylabel('Predicted House Prices');
title('Scatter Plot - Steepest Descent');
legend('show');
grid on;

%% Statistical Analysis of Output

% R-squared and MSE - SGD
Rsquared_SGD = 1 - sum((Y_test - Y_pred_SGD).^2) / sum((Y_test - mean(Y_test)).^2);
MSE_SGD = mean((Y_test - Y_pred_SGD).^2);

disp('Stochastic Gradient Descent Performance:');
disp(['R-squared: ' num2str(Rsquared_SGD)]);
disp(['Mean Squared Error: ' num2str(MSE_SGD)]);

% R-squared and MSE - SD
Rsquared_Steepest = 1 - sum((Y_test - Y_pred_Steepest).^2) / sum((Y_test - mean(Y_test)).^2);
MSE_Steepest = mean((Y_test - Y_pred_Steepest).^2);

fprintf('\n');

disp('Steepest Descent Performance:');
disp(['R-squared: ' num2str(Rsquared_Steepest)]);
disp(['Mean Squared Error: ' num2str(MSE_Steepest)]);
