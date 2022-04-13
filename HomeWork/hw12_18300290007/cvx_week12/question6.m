clc;clear;close all;
m = 100; n = 500; s = 50; 
A = randn(m,n);
xs = zeros(n,1); picks = randperm(n); xs(picks(1:s)) = randn(s,1);
b = A*xs;
tau = 1;
f = @(x) 1/2*norm(A*x - b, 2)^2 + tau*norm(x, 1); % objective function
g = @(x) 1/2*norm(A*x - b, 2)^2; % differentiable part
grad = @(x) A'*(A*x - b); % gradient of g
h = @(x) tau*norm(x, 1); % non-differentiable part
p = @(x,t) prox(x,t); % 不可微部分正交投影
cvx_begin quiet
    variable x(n)
    minimize (0.5*sum_square(A*x-b)+norm(x,1))
cvx_end
f_star = cvx_optval;   % 最优值
% 迭代法输入参数
x0=zeros(n,1);
itermax=1000;
%% fista i)constant ii)line search
subplot(121)
[~,fxlist1]=prox_grad(f,p,grad,1/norm(A'*A),x0,itermax);
[~,fxlist2]=prox_grad_line_search(f,g,p,grad,x0,itermax,0.5);
semilogy(abs(fxlist1-f_star),'linewidth',1);
hold on
semilogy(abs(fxlist2-f_star),'linewidth',1);
hold off
legend('proxGD\_fixed\_step\_size','proxGD\_line\_search');
ylabel('$|f^k-f^*|$','interpreter','latex');
xlabel('iters');
%% fista i)constant ii)line search
subplot(122)
[~,fxlist3]=fista(f,p,grad,1/norm(A'*A),x0,itermax);
[~,fxlist4]=fista_line_search(f,g,p,grad,x0,itermax,0.5);
semilogy(abs(fxlist3-f_star),'linewidth',1);
hold on
semilogy(abs(fxlist4-f_star),'linewidth',1);
hold off
legend('fista\_fixed\_step\_size','fista\_line\_search');
ylabel('$|f^k-f^*|$','interpreter','latex');
xlabel('iters');
%% porx
function y = prox(x,t)
y  = zeros(length(x),1);
ind = find(x>t);
y(ind) = x(ind)- t;
ind = find(x<-t);
y(ind) = x(ind)+ t;
end