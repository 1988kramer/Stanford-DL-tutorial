function [f,g] = softmax_regression(theta, X,y)
    %
    % Arguments:
    %   theta - A vector containing the parameter values to optimize.
    %       In minFunc, theta is reshaped to a long vector.  So we need to
    %       resize it to an n-by-(num_classes-1) matrix.
    %       Recall that we assume theta(:,num_classes) = 0.
    %
    %   X - The examples stored in a matrix.  
    %       X(i,j) is the i'th coordinate of the j'th example.
    %   y - The label for each example.  y(j) is the j'th example's label.
    %
    m=size(X,2);
    n=size(X,1);

    % theta is a vector;  need to reshape to n x num_classes.
    theta=reshape(theta, n, []);
    % add column of zeros to end of theta
    theta = horzcat(theta, zeros(size(theta,1),1));
    num_classes=size(theta,2);

    % initialize objective value and gradient.
    f = 0;
    g = zeros(size(theta));

    %
    % TODO:  Compute the softmax objective function and gradient using vectorized code.
    %        Store the objective function value in 'f', and the gradient in 'g'.
    %        Before returning g, make sure you form it back into a vector with g=g(:);
    %
    %%% YOUR CODE HERE %%%
    
    % compute h(X)
    h = exp(theta' * X); % outputs a matrix of size K x m
    norm = sum(h,1); % computes normalization constants for h
    
    h = bsxfun(@rdivide, h, norm);
    
    % use sub2ind to extract appropriate values from h, then sum
    
    % breaks here with out of range subscript
    % y includes 10s but size(h) = 9xm
    % last column of theta is removed because it's assumed to be 0
    % try padding theta with zeros?
    index = sub2ind(size(h), y, 1:size(h,2));
    
    % compute the cost function
    f = sum(log(h(index))) * -1;
    
    % compute the gradient
    yIndex = zeros(size(h));
    
    yIndex(index) = 1;
    yIndex = yIndex - h;

    g = X * yIndex';
    g = g * -1;
    g = g(:,1:size(g,2) - 1); % remove last column of g
    g=g(:); % make gradient a vector for minFunc
end

