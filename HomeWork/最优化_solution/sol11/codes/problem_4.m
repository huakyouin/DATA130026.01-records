clear;
m = 100; n = 500; s = 50; 
A = randn(m,n);
xs = zeros(n,1); picks = randperm(n); xs(picks(1:s)) = randn(s,1);
b = A*xs;
tau = 0.1;
f = @(x) 1/2*norm(A*x - b, 2)^2 + tau*norm(x, 1); % objective function
g = @(x) 1/2*norm(A*x - b, 2)^2; % differentiable part
grad = @(x) A'*(A*x - b); % gradient of g
h = @(x) tau*norm(x, 1); % non-differentiable part

%% comparison of 2 stepsize

% x0 = zeros(n,1);
% [~, ~, solutionlist1] = prox_grad(A, b, x0, xs, 0.1, 1000);
% [~, ~, solutionlist2] = prox_grad_line_search(A, b, x0, xs, 0.1, 1000);
% 
% semilogy(abs(solutionlist1-f(xs)),'linewidth',2)
% hold on
% semilogy(abs(solutionlist2-f(xs)),'linewidth',2)
% hold off
% legend('proxGD\_fixed\_step\_size','proxGD\_line\_search');
% ylabel('$f^k-f^*$','interpreter','latex');
% xlabel('iteration number','interpreter','latex');


%% performance of lasso under different tau

% x0 = zeros(n,1);
% err_tau_1 = zeros(100, 1);
% err_tau_2 = zeros(100, 1);
% for i = 1:100
% [err_tau_1(i), ~, ~] = prox_grad(A, b, x0, xs, i, 1000);
% [err_tau_2(i), ~, ~] = prox_grad_line_search(A, b, x0, xs, i, 1000);
% end
% 
% plot(1:100, err_tau_1, 'linewidth', 2)
% hold on
% plot(1:100, err_tau_2, 'linewidth', 2)
% hold off
% legend('proxGD\_fixed\_step\_size','proxGD\_line\_search');
% xlabel('$\tau$', 'Interpreter','latex');
% ylabel('$\Vert x - xs \Vert_2$','Interpreter','latex')

%% performance of lasso under different m

% x0 = zeros(n,1);
% err_m_1 = zeros(40, 1);
% err_m_2 = zeros(40, 1);
% for m = 20:20:800
% A_test = randn(m,n);
% b_test = A_test * xs;
% [err_m_1(m/20), ~, ~] = prox_grad(A_test, b_test, x0, xs, 0.1, 1000);
% [err_m_2(m/20), ~, ~] = prox_grad_line_search(A_test, b_test, x0, xs, 0.1, 1000);
% end
% 
% plot(20:20:800, err_m_1, 'linewidth', 2)
% hold on
% plot(20:20:800, err_m_2, 'linewidth', 2)
% hold off
% legend('proxGD\_fixed\_step\_size','proxGD\_line\_search');
% xlabel('$m$', 'Interpreter','latex');
% ylabel('$\Vert x - xs \Vert_2$','Interpreter','latex')
