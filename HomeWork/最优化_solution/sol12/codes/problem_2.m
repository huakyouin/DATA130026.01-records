m = 100;
n = 500;
s = 50;
A = randn(m, n);
xs = zeros(n, 1);
picks = randperm(n);
xs(picks(1:s)) =  rand(s, 1);
b = A*xs;
tau = 1;
f = @(x)0.5*norm(A*x-b)^2 + tau*norm(x, 1);
g = @(x)0.5*norm(A*x-b)^2;
dg = @(x)A'*(A*x-b);
prox = @(x, t)(x-t*tau).*(x>t*tau) + (x+t*tau).*(x<-t*tau);
x0 = zeros(n, 1);

%% cvx
cvx_begin
variable x(500)
minimize 0.5*sum_square(A*x-b)+tau*norm(x, 1)
cvx_end

%% Different algorithms
% Proximal gradient - constant step size
[x, fun_val, gm_val] = proximal_gradient_constant_size(f, dg, prox, x0, 1/max(eig(A'*A)), 1e-6, 1000);
semilogy(10:1000, fun_val(10:1000)-cvx_optval);
hold on

% Proximal gradient - backtracking line search
[x, fun_val, gm_val] = proximal_gradient_line_search(f, g, dg, ...
    prox, x0, 1e-2, 0.5, 1e-6, 1000);
semilogy(10:1000, fun_val(10:1000)-cvx_optval);
hold on

% FISTA - constant step size
[x, fun_val, dg_val] = FISTA_constant_size(f, dg, prox, x0, ...
    1/max(eig(A'*A)), 1e-4, 1000);
semilogy(10:1000, fun_val(10:1000)-cvx_optval);
hold on

% FISTA - line search
[x, fun_val, dg_val] = FISTA_line_search(f, g, dg, prox, x0, ...
    1e-2, 0.5, 1e-4, 1000);
semilogy(10:1000, fun_val(10:1000)-cvx_optval);
legend('proximal constant', 'proximal line search', 'FISTA constant', 'FISTA line search')
xlabel('Iteration')
ylabel('f(x)-f*')
hold off