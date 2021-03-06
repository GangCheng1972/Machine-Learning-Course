function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

%recode the y labels as vectors containing only value 0 or 1
% Y(m, num_labels)
Y = zeros(m, num_labels);
for i = 1:m
    Y(i, y(i)) = 1; 
end

% Calculate the activation function from input X
% z2(m, num_labels)
a1 = [ones(m, 1)  X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

% Calculate the Cost Fuction J without Regularization
J = sum(sum((-Y.*log(a3)-(1.0-Y).*log(1.-a3))))/m;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

% % implement backpropogation algorithm in a loop that processes one example
% % at a time
% for t = 1:m
%     % Step 1: Set the layer value to the t-th training example.
%     a1_s = a1(t,:); 
%     a2_s = a2(t,:); 
%     a3_s = a3(t,:);
%     z2_s = z2(t,:);
%     z3_s = z3(t,:);
%     y_s  =  Y(t,:);
% 
%     % Step 2: Calculate the delta in output layer 3
%     delta_3 = a3_s - y_s;
%     
%     % Step 3: Calculate the delta in hidder layer 2
%     delta_2 = delta_3*Theta2;
%     delta_2 = delta_2(2:end); 
%     delta_2 = delta_2 .* sigmoidGradient(z2_s);
%     
%     % Step 4: Accumulate the gradient
%     Theta2_grad += delta_3'*a2_s; 
%     Theta1_grad += delta_2'*a1_s;
%     
% end

%Calculate the delta in output layer 3 and hidden layer 2
delta_3 = a3 - Y;

delta_2 = delta_3*Theta2; 
delta_2 = delta_2(:,2:end);
delta_2 = delta_2 .* sigmoidGradient(z2);

% Accumulate the gradient and divide by m
Theta2_grad = delta_3'*a2 / m;
Theta1_grad = delta_2'*a1 / m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Calculate the Cost Fuction J with Regularization, excluding the parameter 0
J += lambda*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)))/(2.*m);

% Add regularization into gradient
Theta2_regGrad = lambda*Theta2/m;   
Theta2_regGrad(:,1) = 0;             % should not regularize the parameter 0
Theta2_grad += Theta2_regGrad;

Theta1_regGrad = lambda*Theta1/m;   
Theta1_regGrad(:,1) = 0;             % should not regularize the parameter 0
Theta1_grad += Theta1_regGrad;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
