 %% -----------This is the implemented Steepest Descent method----------- %%
% This function takes training data of x and y values 
% It has the Descent Direction step which involves computing gradient for
% each of the training data (instead of a subset of data in SGD) 
% It has the Update step for theta the parameter, such that as gradient
% decreased, the theta is also minimized

% we use line search method here to help with step size alpha selection as
% we are working with large quantity of data. The line search method
% includes the backtracking method to optimally select a step size

% Some helper functions included are essential, and they are dynamic
% functions for the steepest descent method, making this steepest descent
% implementation more flexible.

function [theta, price_history] = steepestDescentLineSearch(X, y, num_iterations, tol)

    % Initialize parameters theta to zeros
    % - these are which the algorithm will learn during training
    theta = zeros(size(X, 2), 1);

    % Initialize cost history vector to zeros
    % - the algorithm will keep track of how the price differs over the course of training
    % - ideally, at convergence, the last difference should return 0.
    price_history = zeros(num_iterations, 1);

    for iter = 1:num_iterations

        % The Descent Direction step
        % - calculate the gradient of the MSE with respect to theta. 
        gradient = computeGradient(X, y, theta);

        % Use a helper function: lineSearch to find optimal step size
        % - we have large dataset, it is better to avoid using constant
        %   step size for alpha
        % - see helper function defined below
        alpha = lineSearch(X, y, theta, gradient);

        % The Update Step
        % - update parameters theta such that theta moves in opposite of our
        %   gradient
        theta = theta - alpha * gradient;

        % Find the means squared error while we train the data
        % - this is to continuously reduce the mean squared error over the
        %   course of training
        cost = computeCost(X, y, theta);

        % An array to store the prices we have trained
        % - just for our debugging and visualization of how the convergence
        %   work
        price_history(iter) = cost;

        % Check for convergence
        % - if the improvement at current iteration is considered small 
        % (at restol), consider converged and we break the loop and return
        if iter > 1 && abs(price_history(iter) - price_history(iter-1)) < tol
            fprintf('Converged at iteration %d\n', iter);
            break;
        end
    end
end


% -----------------------Helper Functions------------------------%

% Compute gradient for the descent direction step
function gradient = computeGradient(X, y, theta)
  % the number of training data we have
    m = length(y);

    % compute the gradient, in vectorized form, of all and each of the
    % training data
    gradient = (1/m) * X' * (X * theta - y);
end


% Perform line search to find optimal step size using backtracking method
function alpha = lineSearch(X, y, theta, gradient)
    alpha = 1.0;

    % if the MSE at new parameter theta is not sufficiently lower than the
    % current one, we scale the step size alpha
    % this means that if the new MSE is more than the
    % current, we keep reducing alpha until the new MSE is
    % smaller.
    while computeCost(X, y, theta - alpha * gradient) >= computeCost(X, y, theta)
        alpha = alpha / 3;
    end % this continues until a suitable step size is found. 
end

% MSE function
function cost = computeCost(X, y, theta)
    % Calculate the mean squared error of predicted values and actual values
    cost = mean((X * theta - y).^2);
end
