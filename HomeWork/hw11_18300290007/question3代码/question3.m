%% problem 4
% 生成数据
m = 100;
n = 500;
s = 50;
A = randn(m,n);
xs = zeros(n,1);
picks = randperm(n);
xs(picks(1:s)) = randn(s,1);
b = A * xs;
tau = 0.001;
% 原函数及梯度函数
func = @(x) norm(A*x-b)^2/2 + tau*norm(x,1);
grad = @(x) lasso_subg(x,A,b,tau);
% CVX求解
cvx_begin quiet
    variable x(n)
    minimize(sum_square(A*x-b)/2 + tau*norm(x,1))
cvx_end
f_star = cvx_optval;   % 最优值
% 迭代法数据定义
itermax = 10000; 
x0 = zeros(n,1);  

%% constant step size
subplot(2,2,1)
% 步长1e-3
[funclist, ~] = subgradient_method(func, grad, x0, itermax, 1, 1e-3);
x = 1:itermax;
y = abs(funclist - f_star);
semilogy(x,y);
hold on
% 步长1e-4
[funclist, ~] = subgradient_method(func, grad, x0, itermax, 1, 1e-4);
x = 1:itermax;
y = abs(funclist - f_star);
semilogy(x,y);
% 步长1e-5
[funclist, ~] = subgradient_method(func, grad, x0, itermax, 1, 1e-5);
x = 1:itermax;
y = abs(funclist - f_star);
semilogy(x,y);
% 图注释
title('constant step size')
xlabel('Iters')
ylabel('$|f(x_k)-f^*|$','Interpreter','latex')
h = legend('$10^{-3}$','$10^{-4}$','$10^{-5}$');
set(h,'Interpreter','latex')
hold off

%% constant step length
subplot(2,2,2)
%偏移距离1e-1
[funclist, ~] = subgradient_method(func, grad, x0, itermax, 2, 1e-1);
x = 1:itermax;
y = abs(funclist - f_star);
semilogy(x,y);
hold on
%偏移距离1e-2
[funclist, ~] = subgradient_method(func, grad, x0, itermax, 2, 1e-2);
x = 1:itermax;
y = abs(funclist - f_star);
semilogy(x,y);
%偏移距离1e-3
[funclist, ~] = subgradient_method(func, grad, x0, itermax, 2, 1e-3);
x = 1:itermax;
y = abs(funclist - f_star);
semilogy(x,y);
% 图注释
title('constant step length')
xlabel('Iters')
ylabel('$|f(x_k)-f^*|$','Interpreter','latex')
h = legend('$10^{-1}/\|g_k\|$','$10^{-2}/\|g_k\|$','$10^{-3}/\|g_k\|$');
set(h,'Interpreter','latex')
hold off

%% diminishing step size
subplot(2,2,3)
% 递减步长系数1e-2
[funclist, ~] = subgradient_method(func, grad, x0, itermax, 3, 1e-2);
x = 1:itermax;
y = abs(funclist - f_star);
semilogy(x,y);
hold on
% 递减步长系数1e-3
[funclist, ~] = subgradient_method(func, grad, x0, itermax, 3, 1e-3);
x = 1:itermax;
y = abs(funclist - f_star);
semilogy(x,y);
% 递减步长系数1e-4
[funclist, ~] = subgradient_method(func, grad, x0, itermax, 3, 1e-4);
x = 1:itermax;
y = abs(funclist - f_star);
semilogy(x,y);
% 图注释
title('diminishing step size')
xlabel('Iters')
ylabel('$|f(x_k)-f^*|$','Interpreter','latex')
h = legend('$10^{-2}/\sqrt{k}$','$10^{-3}/\sqrt{k}$','$10^{-4}/\sqrt{k}$');
set(h,'Interpreter','latex')
hold off

%% Polyak's step size
subplot(2,2,4)
[funclist,~] = subg_method_polyak(func, grad, x0, itermax, f_star);
x = 1:itermax;
y = abs(funclist - f_star);
semilogy(x,y);
% 图注释
title('Polyak''s step size')
xlabel('Iters')
ylabel('$|f(x_k)-f^*|$','Interpreter','latex')
