%% problem 2(a)

% function, gradient and Hessian
func = @(x) exp(x(1)+3*x(2)-0.1)+exp(x(1)-3*x(2)-0.1)+exp(-x(1)-0.1)+0.1*(x(1)^2+x(2)^2);
grad = @(x) [exp(x(1)+3*x(2)-0.1)+exp(x(1)-3*x(2)-0.1)-exp(-x(1)-0.1)+0.2*(x(1));...
     3*exp(x(1)+3*x(2)-0.1)-3*exp(x(1)-3*x(2)-0.1)+0.2*(x(2))];
hess = @(x)  [exp(x(1)+3*x(2)-0.1)+exp(x(1)-3*x(2)-0.1)+exp(-x(1)-0.1)+0.2, 3*exp(x(1)+3*x(2)-0.1)-3*exp(x(1)-3*x(2)-0.1);
     3*exp(x(1)+3*x(2)-0.1)-3*exp(x(1)-3*x(2)-0.1), 9*exp(x(1)+3*x(2)-0.1)+9*exp(x(1)-3*x(2)-0.1)+0.2];

tol = 1e-7;     % convergence tolerance

%% the first graph

subplot(1,2,1)

x0 = [1;1];     % the first initial point

% gradient descent method
tic;
[funclist, gradlist, sol_GD] = GD(func, grad, x0, tol);
t_GD = toc;
maxit = length(find(funclist~=0));
x = 1:maxit;
y = gradlist(1:maxit);
semilogy(x,y,'-*','LineWidth',2);
hold on

% damped Newton's method
tic;
[funclist, gradlist, sol_Newton] = Newton(func, grad, hess, x0, tol);
t_Newton = toc;
maxit = length(find(funclist~=0));
x = 1:maxit;
y = gradlist(1:maxit);
semilogy(x,y,'-*','LineWidth',2);

title('$x_0=(1,1)^T$','Interpreter','latex')
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

x0 = [-1;-1];       % the second initial point

% gradient descent method
tic;
[funclist, gradlist, sol_GD] = GD(func, grad, x0, tol);
t_GD = toc;
maxit = length(find(funclist~=0));
x = 1:maxit;
y = gradlist(1:maxit);
semilogy(x,y,'-*','LineWidth',2);
hold on

% damped Newton's method
tic;
[funclist, gradlist, sol_Newton] = Newton(func, grad, hess, x0, tol);
t_Newton = toc;
maxit = length(find(funclist~=0));
x = 1:maxit;
y = gradlist(1:maxit);
semilogy(x,y,'-*','LineWidth',2);

title('$x_0=(-1,-1)^T$','Interpreter','latex')
xlabel('Iteration number')
ylabel('$\Vert\nabla f(x)\Vert$','Interpreter','latex')
legend('GD','Newton')
hold off

% print the optimal values and times of algorithms
Algorithm = {'GD';'Newton'};
Optimal_value = [func(sol_GD);func(sol_Newton)];
Time = [t_GD;t_Newton;];
table(Algorithm, Optimal_value, Time)
