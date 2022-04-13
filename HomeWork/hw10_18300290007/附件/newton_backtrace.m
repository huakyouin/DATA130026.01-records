%------------牛顿+A回溯迭代法
%-----
%f,J,H分别为原函数，梯度和黑塞阵
%err为容忍误差
%alpha，beta为A回溯参数
%output为v则输出最优值，vs则输出最优解+最优值
%-----
function normJf=newton_backtrace(f,J,H,x,err,alpha,beta,output)
% 初始化
normJf = [];
i = 1;
while 1
    % 空间换时间
    hval=H(x);
    jval=J(x);
    % 确定迭代方向
    direction = -hval \ jval;
    t = 1;% tk 的初始值
    % 回溯部分
    while f(x) - f(x+t*direction) < -alpha * t * jval.' * direction
       t = beta * t;
    end
    % 更新
    if f(x)<f(x+t*direction) disp(i); end % 输出迭代后反而函数值增加的回合数，用于检查迭代正确性
    x=x+t*direction;
    normJf(i) = norm(jval);
    % 停止条件
    if normJf(i) <= err
        break
    end
    i = i + 1;
end
opt_x_newton = x';
opt_solution_newton = f(x);
% 是否输出问题结果
if output=='v' display(opt_solution_newton); end
if output=='vs' display(opt_x_newton);display(opt_solution_newton); end
end