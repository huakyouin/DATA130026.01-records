% 次梯度迭代之Polyak's step size
%----------输入参数
% - func: 原函数
% - grad: 梯度函数
% - x0: 迭代起点
% - itermax: 最大迭代步数
% - f_star: 问题最优值
%----------
%----------输出说明
% - funclist: 每代函数值
% - solution: 算法给出最优解（未必真最优）
%----------
function [funclist,solution] = SGD_Polyak(func, grad, x0, itermax, f_star)
funclist = zeros(1,itermax);
% 迭代初始化
x = x0;
funclist(1) = func(x);
for iter = 2:itermax
    direction = -grad(x); %下降方向
    % Polyak's步长
    t = (func(x)-f_star)/norm(direction)^2;  
    % 更新
    x = x + t*direction;
    funclist(iter) = func(x);
end
solution = x;

end

