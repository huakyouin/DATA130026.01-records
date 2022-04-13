function hess = LRhess(A,b,w)
% the Hessian of the objective function of logistic regression problem

[n,m]=size(A);
sumh = zeros(n, n);
for i = 1:m
    temp = exp(-b(i) * (w' * A(:, i)));
    sumh = sumh + temp / ((1+temp)^2) * b(i)^2 * (A(:, i) * A(:, i)');
end
hess = 0.02 * eye(n) + (1/m)*sumh;
