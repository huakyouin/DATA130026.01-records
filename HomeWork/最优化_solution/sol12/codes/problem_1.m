m = 500;
n = 1000;
A = randn(n, m);
b = sign(rand(m, 1) - 0.5);

J = @(w, c)mean(log(1+exp(-b.*(A'*w+c)))) + 0.01*(w'*w+c^2);
dw = @(w, c)-A*(b./(1+exp(b.*(A'*w+c))))/m + 0.02*w;
dc = @(w, c)-1*sum(b./(1+exp(b.*(A'*w+c))))/m + 0.02*c;
% J = @(w, c)mean(log(1+exp(-b.*(A'*w+c))));
% dw = @(w, c)-A*(b./(1+exp(b.*(A'*w+c))))/m;
% dc = @(w, c)-1*sum(b./(1+exp(b.*(A'*w+c))))/m;

f = @(x)J(x(1:n), x(n+1));
g = @(x)[dw(x(1:n), x(n+1)); dc(x(1:n), x(n+1))];
prox = @(x, t)x;
w0 = zeros(n, 1);
c0 = 0;
x0 = [w0; c0];

%% Different algorithms
% Gradient method
[x, funv, grad] = gradient_method_constant(f, g, x0, 5e-3, 1e-8, 2000);
semilogy(1:length(grad), grad(1:length(grad)));
hold on

% FISTA
[x, fun_val, grad] = FISTA_constant_size(f, g, prox, x0, ...
    5e-3, 1e-8, 2000);
semilogy(1:length(grad), grad(1:length(grad)));
hold on

% FISTA descent
[x, fun_val, grad] = FISTA_descent(f, g, prox, x0, ...
    5e-3, 1e-8, 2000);
semilogy(1:length(grad), grad(1:length(grad)));
hold on

% FISTA restart 100
[x, fun_val, grad] = FISTA_restart_T(f, g, prox, x0, ...
    5e-3, 1e-8, 2000, 100);
semilogy(1:length(grad), grad(1:length(grad)), '--');
hold on

% FISTA restart OCf
[x, fun_val, grad] = FISTA_restart_OCf(f, g, prox, x0, ...
    5e-3, 1e-8, 2000);
semilogy(1:length(grad), grad(1:length(grad)), '--');
hold on

% FISTA restart OCg
[x, fun_val, grad] = FISTA_restart_OCg(f, g, prox, x0, ...
    5e-3, 1e-8, 2000);
semilogy(1:length(grad), grad(1:length(grad)), '--');
hold on
legend('gradient', 'fista', 'descent fista', 'restart fista (100)',...
    'restart fista (OCf)', 'restart fista (OCg)')
xlabel('Iteration')
ylabel('grad norm')
hold off