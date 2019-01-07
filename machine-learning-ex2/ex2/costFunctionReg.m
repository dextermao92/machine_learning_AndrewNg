function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = X*theta;
part1 = -y.*log(sigmoid(z));
part2 = -(1.-y).*log(1.-sigmoid(z));
J1 = sum(part1+part2)/(m); 
J2 = lambda.*sum(theta(2:end,1).^2)/(2*m);

J = J1+J2;

% update gradient
% gradient: theta0
grad(1) = sum(X(:,1).*(sigmoid(z) - y))/m;

% gradient: theta1 ~ n
theta_size = size(theta);
for i = 2:theta_size
    grad(i) = ((sum(X(:,i).*(sigmoid(z) - y))) + sum(lambda.*theta(i)))/m;
end 

% =============================================================

end
