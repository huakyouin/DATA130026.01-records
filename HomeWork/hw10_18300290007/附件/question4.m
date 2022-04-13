%% 4
clear all;
% 生成系数
rc = 1:10:1000;
A = sprandsym(100,0.1,rc);
b = randn(100,1);
% F(x)
f = @(x) x' * A * x / 2 + b' * x;
% 直接获得最优解
optx = -A\b;
optf = f(optx);
% F'(x)
J = @(x) A * x + b;
% hessa矩阵初始化:用A逆的对角元素
H = diag(diag(inv(A))); 
% 可视化
fg3 = figure('numbertitle','off','name','第4题');
subplot(121)
plot(log10(BFGS_Hk_backtrace(f,J,H,optf,1e-6,0.5,0.9,'v')));title('BFGS');ylabel('log_1_0（fk-f*）');xlabel('iters')
subplot(122)
plot(log10(BB_backtrace(f,J,optf,1e-6,0.5,0.9,'v')));title('BB method');ylabel('log_1_0（fk-f*）');xlabel('iters')
sgtitle('4')