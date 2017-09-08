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
    a{a + l} = 1 ./ (1 + exp(-z)); % sigmoid function
    % a{l + 1} = bsxfun(@rdivide, exp(z), sum(exp(z), 1)); % softmax activation function
end

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
yMatrix = zeros(ei.output_dim, size(data, 2));
for i= 1:size(data, 2)
    yMatrix(labels(i), i) = 1;
end
hwb = a{numel(layer_sizes) + 1};
prob = bsxfun(@rdivide, exp(hwb), exp(sum(hwb, 1)));
h = yMatrix .* log(prob);
cost = -1 * sum(sum(h,1));

%% compute gradients using backpropagation
% make delta matrices
deltas = cell(1, numel(layer_sizes) + 1);
% compute delta for output layer
deltas{numel(layer_sizes) + 1} = -1 * sum(yMatrix - prob, 2);
% compute deltas for hidden layers
for i = numel(layer_sizes):1
    deltas{i} = stack{i}.W' * deltas{i + 1} .* (a{i}(1 - a{i}));
end
for i = 1:numHidden + 1
    gradStack{i}.W = deltas{i + 1} * a{i}';
    gradStack{i}.b = deltas{i + 1};
end
%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



