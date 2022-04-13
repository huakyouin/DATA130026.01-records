% 计算LASSO问题次梯度的函数
function grad = lasso_subg(x, A, b, tau)
grad = A'*(A*x-b);
grad(x>0) = grad(x>0) + tau;
grad(x<0) = grad(x<0) - tau;
end

