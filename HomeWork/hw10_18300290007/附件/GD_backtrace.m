%------------梯度下降+A回溯法
%-----
%f,J,H分别为原函数，梯度和黑塞阵
%err为容忍误差
%alpha，beta为A回溯参数
%output为v则输出最优值，vs则输出最优解+最优值
%-----
function normJf2=GD_backtrace(f,J,x,err,alpha,beta,output)
normJf2 = [];
i = 1;
while 1
    % 空间换时间
    jval=J(x);
    % 以梯度作为下降方向
    direction = double(-jval);
    t = 1;% tk 的初始值
    % 回溯部分
    while double(f(x)) - double(f(x+t*direction)) < double(-alpha * t * jval.' * direction)
       t = beta * t;
    end
    % 更新
    if f(x)<f(x+t*direction) disp(i); end % 输出迭代后反而函数值增加的回合数，用于检查迭代正确性
    x=x+t*direction;
    
    % 停止判断
    normJf2(i) = norm(jval);
    if normJf2(i) <= err
        break
    end
    i = i + 1;
end
opt_x_GD = x';
opt_solution_GD = f(x);
% 是否输出问题结果
if output=='v' display(opt_solution_GD); end
if output=='vs' display(opt_x_GD);display(opt_solution_GD); end
end