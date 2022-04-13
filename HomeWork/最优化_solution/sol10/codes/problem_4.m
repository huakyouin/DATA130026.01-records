%% problem 4

% rng(222)    % rand seed

% generate the data
m = 100;
n = 500;
s = 50;
A = randn(m,n);
xs = zeros(n,1);
picks = randperm(n);
xs(picks(1:s)) = randn(s,1);
b = A * xs;
tau = 0.001;

% function and subgradient
func = @(x) norm(A*x-b)^2/2 + tau*norm(x,1);
grad = @(x) LASSO_subgrad(x,A,b,tau);

% compute the optimal value by CVX
cvx_begin quiet
    variable x(n)
    minimize(sum_square(A*x-b)/2 + tau*norm(x,1))
cvx_end
f_star = cvx_optval;   % the optimal value

itermax = 10000;    % iteration number
x0 = zeros(n,1);    % initial point

%% constant step size

subplot(2,2,1)

[funclist, ~, ~] = SGD(func, grad, x0, itermax, 1, 1e-3);
x = 1:itermax;
y = abs(funclist - f_star);
semilogy(x,y,'-','LineWidth',2);
hold on

[funclist, ~, ~] = SGD(func, grad, x0, itermax, 1, 1e-4);
x = 1:itermax;
y = abs(funclist - f_star);
semilogy(x,y,'-','LineWidth',2);

[funclist, ~, ~] = SGD(func, grad, x0, itermax, 1, 1e-5);
x = 1:itermax;
y = abs(funclist - f_star);
semilogy(x,y,'-','LineWidth',2);

title('constant step size')
xlabel('Iteration number')
ylabel('$|f(x_k)-f^*|$','Interpreter','latex')
h = legend('$10^{-3}$','$10^{-4}$','$10^{-5}$');
set(h,'Interpreter','latex')
hold off

%% constant step length

subplot(2,2,2)

[funclist, ~, ~] = SGD(func, grad, x0, itermax, 2, 1e-1);
x = 1:itermax;
y = abs(funclist - f_star);
semilogy(x,y,'-','LineWidth',2);
hold on

[funclist, ~, ~] = SGD(func, grad, x0, itermax, 2, 1e-2);
x = 1:itermax;
y = abs(funclist - f_star);
semilogy(x,y,'-','LineWidth',2);

[funclist, ~, ~] = SGD(func, grad, x0, itermax, 2, 1e-3);
x = 1:itermax;
y = abs(funclist - f_star);
semilogy(x,y,'-','LineWidth',2);

title('constant step length')
xlabel('Iteration number')
ylabel('$|f(x_k)-f^*|$','Interpreter','latex')
h = legend('$10^{-1}/\|g_k\|$','$10^{-2}/\|g_k\|$','$10^{-3}/\|g_k\|$');
set(h,'Interpreter','latex')
hold off

%% diminishing step size

subplot(2,2,3)

[funclist, ~, ~] = SGD(func, grad, x0, itermax, 3, 1e-2);
x = 1:itermax;
y = abs(funclist - f_star);
semilogy(x,y,'-','LineWidth',2);
hold on

[funclist, ~, ~] = SGD(func, grad, x0, itermax, 3, 1e-3);
x = 1:itermax;
y = abs(funclist - f_star);
semilogy(x,y,'-','LineWidth',2);

[funclist, ~, ~] = SGD(func, grad, x0, itermax, 3, 1e-4);
x = 1:itermax;
y = abs(funclist - f_star);
semilogy(x,y,'-','LineWidth',2);

title('diminishing step size')
xlabel('Iteration number')
ylabel('$|f(x_k)-f^*|$','Interpreter','latex')
h = legend('$10^{-2}/\sqrt{k}$','$10^{-3}/\sqrt{k}$','$10^{-4}/\sqrt{k}$');
set(h,'Interpreter','latex')
hold off

%% Polyak's step size

subplot(2,2,4)

[funclist, ~, ~] = SGD_Polyak(func, grad, x0, itermax, f_star);
x = 1:itermax;
y = abs(funclist - f_star);
semilogy(x,y,'-','LineWidth',2);
title('Polyak''s step size')
xlabel('Iteration number')
ylabel('$|f(x_k)-f^*|$','Interpreter','latex')
