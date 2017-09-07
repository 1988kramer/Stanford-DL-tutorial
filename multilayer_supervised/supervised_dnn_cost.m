function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1); % activation functions
gradStack = cell(numHidden+1, 1); % gradient stack
%% forward propagation
a = cell(1, numel(ei.layer_sizes) + 1); % should store the results of the 
                                        % activation functions for each layer
a{1} = data; % activations for layer 1 are just the inputs

for l = 1:numel(ei.layer_sizes)
    z = stack{l}.W * a{l} + stack{l}.b;
    a{l + 1} = bsxfun(@rdivide, exp(z), sum(exp(z), 1)); % softmax activation function
end

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



