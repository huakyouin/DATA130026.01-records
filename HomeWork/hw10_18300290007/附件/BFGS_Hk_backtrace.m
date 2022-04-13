% ------------基于H^k的BFGS法
%-----
%f,J,H分别表示目标函数，其梯度和黑塞阵
%optf为理论最优值
%err为容忍误差
%alpha，beta为A回溯参数
%output为v则输出最优值，vs则输出最优解+最优值
%-----
function err_f=BFGS_Hk_backtrace(f,J,H,optf,err,alpha,beta,output)
% 初始化
I = eye(100);
xk = ones(100,1);
%回溯设置
i = 1; % 迭代次数记录
err_f = [];
while 1
    % 计算方向
    Jk = J(xk);
    dk = -H * Jk;
    t = 1; % tk 的初始值
    % 回溯
    while f(xk) - f(xk + t * dk) < -alpha * t * (Jk') * dk
       t = beta * t;
    end
    % 更新
    err_f(i) =abs(f(xk) - optf);
    xk = xk + t * dk;
    s = t * dk;
    y = J(xk + t * dk) - J(xk);
    ro = 1 / (y' * s);
    H = (I - ro * s * (y')) * H * (I - ro * y * (s')) + ro * s * (s');
    % 停止条件
    if norm(Jk) < err || i>=1000
        break
    end
    i = i + 1;
end
opt_x_BFGS = xk';
opt_solution_BFGS = f(xk);
% 是否输出问题结果
if output=='v' display(opt_solution_BFGS); end
if output=='vs' display(opt_x_BFGS);display(opt_solution_BFGS); end
end