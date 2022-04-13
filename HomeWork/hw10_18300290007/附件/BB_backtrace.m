% ---------------Barzilai-Borwein 梯度法
%-----
%f,J,H分别表示目标函数，其梯度和黑塞阵
%optf为理论最优值
%err为容忍误差
%alpha，beta为A回溯参数
%output为v则输出最优值，vs则输出最优解+最优值
%-----
function err_f2=BB_backtrace(f,J,optf,err,alpha,beta,output)
% 初始化
i = 1;
t=1;
%回溯设置
err_f2=[];
xk = ones(100,1);
y=J(xk);
s = t * J(xk);
while 1
    % 以梯度作为下降方向
    dk = -J(xk);
    % BB
    t = s'*y/(y'*y);% tk 的初始值
    % 回溯
    while f(xk) - f(xk + t * dk) < -alpha * t * dk' * dk
       t = beta * t;
    end
    % 更新
    err_f2(i)=f(xk)-optf;
    s = t * dk;
    y = J(xk + t * dk) - J(xk);
    xk=xk+t*dk;
    % 停止条件
    if norm(dk) < err || i>=1000
        break
    end
    i = i + 1;
end
opt_x_BB = xk';
opt_solution_BB = f(xk);
% 是否输出问题结果
if output=='v' display(opt_solution_BB); end
if output=='vs' display(opt_x_BB);display(opt_solution_BB); end
end