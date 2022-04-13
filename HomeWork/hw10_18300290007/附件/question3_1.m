%% 3-1 
clc;clear all;close all;
syms x1 x2;
%生成问题
f = @(x1, x2) exp(x1 + 3 * x2 - 0.1) + exp(x1 - 3 * x2 - 0.1) + exp(-x1 - 0.1) + 0.1 * (x1 * x1 + x2 * x2);
J = @(x1, x2) [x1/5 + exp(x1 - 3*x2 - 1/10) + exp(x1 + 3*x2 - 1/10) - exp(- x1 - 1/10), x2/5 - 3*exp(x1 - 3*x2 - 1/10) + 3*exp(x1 + 3*x2 - 1/10)].';
H = @(x1, x2) [exp(x1 - 3*x2 - 1/10) + exp(x1 + 3*x2 - 1/10) + exp(- x1 - 1/10) + 1/5,       3*exp(x1 + 3*x2 - 1/10) - 3*exp(x1 - 3*x2 - 1/10)
          3*exp(x1 + 3*x2 - 1/10) - 3*exp(x1 - 3*x2 - 1/10), 9*exp(x1 - 3*x2 - 1/10) + 9*exp(x1 + 3*x2 - 1/10) + 1/5];
f=@(x) f(x(1),x(2));
J=@(x) J(x(1),x(2));
H=@(x) H(x(1),x(2));
% 可视化
fg1 = figure('numbertitle','off','name','第3题（a）');
subplot(121)
semilogy(newton_backtrace(f,J,H,[1;1],1e-7,0.5,0.5,'vs'));title('牛顿法');ylabel('||JF||_2');xlabel('iters')
subplot(122)
semilogy(GD_backtrace(f,J,[1;1],1e-7,0.5,0.5,'vs'));title('梯度法');ylabel('||JF||_2');xlabel('iters')
sgtitle('3-1')