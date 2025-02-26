% The objective function is the minimization problem of the mean squared error

% The function takes in theta, X, y
% Interpretation: this function is the mean squared error
% of the predicted housing prices and actual value prices.

% ----------------------- Definitions of Parameters --------------------%
% theta = model parameter; coefficients to our linear model. 

% X is the input matrix of the selected features (only 5 numerical
% features selected, based on their statistical significance)

% y is the response variable which is housing price. 
%-----------------------------------------------------------------------%

function mse = findMSE(theta, X, y)
    m = length(y);
    predictions = X * theta; % parameterizing our features 
    squaredErrors = (predictions - y).^2; % taking the difference
    mse = (1/(2*m)) * sum(squaredErrors); % finding the mean of the difference
end