function func = LRfunc(A,b,w)
% the objective function of logistic regression problem

[n,m] = size(A);
sumitem = zeros(m,1);
for i = 1:m
    sumitem(i) = log(1 + exp(-b(i)*(w'*A(:,i))));
end
func = mean(sumitem) + 0.01*(w'*w);