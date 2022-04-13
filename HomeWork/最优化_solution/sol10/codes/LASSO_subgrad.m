% 计算LASSO问题次梯度的函数
function grad = LASSO_subgrad(x, A, b, tau)
grad = A'*(A*x-b);
grad(x>0) = grad(x>0) + tau;
grad(x<0) = grad(x<0) - tau;
% grad(x==0) = grad(x==0) + rand(length(find(x==0)),1)*tau;

end

