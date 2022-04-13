function grad = LRgrad(A,b,w)
% the gradient of the objective function of logistic regression problem

[n,m]=size(A);
sumg = zeros(n, 1);
for i = 1:m
    sumg = sumg + (1 - 1 / (1+exp(-b(i) * (w' * A(:, i))))) * -b(i) * A(:, i);
end
grad = 0.02*w + (1/m)*sumg;

