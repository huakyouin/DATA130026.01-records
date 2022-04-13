%% problem 2(b)

% rng(9)        % rand seed

% function, gradient and Hessian
m = 500; n = 1000;
A = randn(n,m); 
b = sign(rand(m,1)-0.5);
A = [A',ones(m,1)]';
func = @(w)LRfunc(A,b,w);
grad = @(w)LRgrad(A,b,w);
hess = @(w)LRhess(A,b,w);

tol = 1e-4;     % convergence tolerance

%% the first graph

subplot(1,2,1)

x0 = ones(n+1,1);   % the first initial point

% gradient descent method
tic;
[funclist, gradlist, sol_GD] = GD(func, grad, x0, tol);
t_GD = toc;
maxit = length(find(funclist~=0));
x = 1:maxit;
y = gradlist(1:maxit);
semilogy(x,y,'-','LineWidth',2);
hold on

% damped Newton's method
tic;
[funclist, gradlist, sol_Newton] = Newton(func, grad, hess, x0, tol);
t_Newton = toc;
maxit = length(find(funclist~=0));
x = 1:maxit;
y = gradlist(1:maxit);
semilogy(x,y,'-+','LineWidth',2);

title('$x_0=(1,\dots,1)^T$','Interpreter','latex')
xlabel('Iteration number')
ylabel('$\Vert\nabla f(x)\Vert$','Interpreter','latex')
legend('GD','Newton')
hold off

% print the optimal values and times of algorithms
Algorithm = {'GD';'Newton'};
Optimal_value = [func(sol_GD);func(sol_Newton)];
Time = [t_GD;t_Newton];
table(Algorithm, Optimal_value, Time)

%% the second graph

subplot(1,2,2)

x0 = zeros(n+1,1);      % the second initial point

% gradient descent method
tic;
[funclist, gradlist, sol_GD] = GD(func, grad, x0, tol);
t_GD = toc;
maxit = length(find(funclist~=0));
x = 1:maxit;
y = gradlist(1:maxit);
semilogy(x,y,'-','LineWidth',2);
hold on

% damped Newton's method
tic;
[funclist, gradlist, sol_Newton] = Newton(func, grad, hess, x0, tol);
t_Newton = toc;
maxit = length(find(funclist~=0));
x = 1:maxit;
y = gradlist(1:maxit);
semilogy(x,y,'-+','LineWidth',2);

title('$x_0=(0,\dots,0)^T$','Interpreter','latex')
xlabel('Iteration number')
ylabel('$\Vert\nabla f(x)\Vert$','Interpreter','latex')
legend('GD','Newton')
hold off

% print the optimal values and times of algorithms
Algorithm = {'GD';'Newton'};
Optimal_value = [func(sol_GD);func(sol_Newton)];
Time = [t_GD;t_Newton];
table(Algorithm, Optimal_value, Time)

