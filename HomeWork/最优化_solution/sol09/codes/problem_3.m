%% problem 3

% rng(555)        % rand seed

% function, gradient and Hessian
rc = 1:10:1000;
A = sprandsym(100,0.1,rc);
b = randn(100,1);
func = @(x) x'*A*x/2+b'*x;
grad = @(x) A*x+b;
hess = @(x) A;

n = 100;            % dimension
x0 = ones(n,1);     % initial point
fstar = func(-A\b); % the optimal value
tol = 1e-6;         % convergence tolerance

% BFGS
tic;
[funclist, ~, sol_BFGS] = BFGS(func, grad, x0, tol);
t_BFGS = toc;
maxit = length(find(funclist~=0));
x = 1:maxit;
y = abs(funclist(1:maxit) - fstar);
semilogy(x,y,'-','LineWidth',2);
hold on

% BB step GD (1st update rule)
update = 1;
tic;
[funclist, ~, sol_BBGD1] = BBGD(update, func, grad, x0, tol);
t_BBGD1 = toc;
maxit = length(find(funclist~=0));
x = 1:maxit;
y = abs(funclist(1:maxit) - fstar);
semilogy(x,y,'-','LineWidth',2);

% BB step GD (2nd update rule)
update = 2;
tic;
[funclist, ~, sol_BBGD2] = BBGD(update, func, grad, x0, tol);
t_BBGD2 = toc;
maxit = length(find(funclist~=0));
x = 1:maxit;
y = abs(funclist(1:maxit) - fstar);
semilogy(x,y,'-','LineWidth',2);

title('$x_0=(1,\dots,1)^T$','Interpreter','latex')
xlabel('Iteration number')
ylabel('$\vert f(x) - f^*\vert$','Interpreter','latex')
legend('BFGS','BB-1','BB-2')
hold off

% print the optimal values and times of algorithms
Algorithm = {'BFGS';'BB-1';'BB-2'};
Optimal_value = [func(sol_BFGS);func(sol_BBGD1);func(sol_BBGD2)];
Time = [t_BFGS;t_BBGD1;t_BBGD2];
table(Algorithm, Optimal_value, Time)

