function [f,g] = logistic_regression(theta, X,y)
    %
    % Arguments:
    %   theta - A column vector containing the parameter values to optimize.
    %   X - The examples stored in a matrix.  
    %       X(i,j) is the i'th coordinate of the j'th example.
    %   y - The label for each example.  y(j) is the j'th example's label.
    %

    m=size(X,2);
    n = size(X,1);

    % initialize objective value and gradient.
    f = 0;
    g = zeros(size(theta));


    %
    % TODO:  Compute the objective function by looping over the dataset and summing
    %        up the objective values for each example.  Store the result in 'f'.
    %
    % TODO:  Compute the gradient of the objective by looping over the dataset and summing
    %        up the gradients (df/dtheta) for each example. Store the result in 'g'.
    %
    %%% YOUR CODE HERE %%%
    
    % calculate h(x), the result of the sigmoid function
    h = 1 ./ (1 + exp(-theta' * X));
    
    % calculate f, the cost function
    for i = 1:m
        f = f + ((y(i) * log(h(i))) + ((1 - y(i)) * log(1 - h(i))));
    end
    f = -f;
    
    % calculate g, the gradient function
    for j = 1:n
        for i = 1:m
            g(j) = g(j) + (X(j,i) * (h(i) - y(i)));
        end
    end 
end