% 次梯度迭代法（多类一体）
%----------输入参数
% - func: 原函数
% - grad: 梯度函数
% - x0: 迭代起点
% - itermax: 最大迭代步数
% - rule: 迭代法类型
%         1 : 固定步长
%         2 : 固定偏移距离
%         3 : 递减步长
% - para: 步长设定
%----------
%----------输出说明
% - funclist: 每代函数值
% - solution: 算法给出最优解（未必真最优）
%----------
function [funclist,solution] = subgradient_method(func, grad, x0, itermax, rule, para)
funclist = zeros(1,itermax);
% 初始化
x = x0;
funclist(1) = func(x);
for iter = 2:itermax
    direction = -grad(x); % 下降方向
    % 计算步长t
    if rule == 1            % 固定步长
        t = para;
    elseif rule == 2        % 固定偏移距离
        t = para/norm(direction);
    elseif rule == 3        % 递减步长
        t = para/sqrt(iter);
    end
    % 更新
    x = x + t*direction;
    funclist(iter) = func(x);
end
solution = x;
end

