clc;clear all;close all
% 生成数据
n = 500;
xbar=randn(n,1);
Q = randn(n,n);
Q=Q*Q';
Q=Q+Q'+eye(n);
b=Q*xbar;
e=ones(n,1);
% 原函数及梯度函数
f = @(x) 0.5*x'*Q*x-b'*x+1e10*ones(1,n)*(x>2|x<1);
p = @(x,t) x.*(x>=1 &x<=2)+1*(x<1)+2*(x>2); % 不可微部分正交投影
g = @(x) 0.5*x'*Q*x-b'*x; % 可微部分函数
grad = @(x) Q*x-b;
% cvx求解最优值
cvx_begin quiet
    variable x(n)
    minimize (0.5*x'*Q*x-b'*x)
        subject to
            x>=1
            x<=2
cvx_end
f_star = cvx_optval;   % 最优值
% 迭代法输入参数
itermax = 1000; 
x0 = e;  
%% prox_grad i)cosntant ii)line search
subplot(121)
[~,fxlist1]=prox_grad(f,p,grad,1/norm(Q),x0,itermax);
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
[~,fxlist3]=fista(f,p,grad,1/norm(Q),x0,itermax);
[~,fxlist4]=fista_line_search(f,g,p,grad,x0,itermax,0.5);
semilogy(abs(fxlist3-f_star),'linewidth',1);
hold on
semilogy(abs(fxlist4-f_star),'linewidth',1);
hold off
legend('fista\_fixed\_step\_size','fista\_line\_search');
ylabel('$|f^k-f^*|$','interpreter','latex');
xlabel('iters');