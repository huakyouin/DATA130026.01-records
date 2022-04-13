% 投影梯度法  step size=1/L
%----------输入参数
% - f: 原函数
% - p：函数不可微部分的正交投影函数
% - grad：函数可微部分梯度
% - fixt:固定步长
% - x0: 迭代起点
% - itermax: 最大迭代步数
%----------
%----------输出说明
% - fxlist: 每步迭代函数值列表
% - x_end: 算法给出最优解（未必真最优）
%----------
function [x_end, fxlist] = prox_grad(f,p,grad,fixt,x0,itermax)
tol = 1e-6; % 可以接受的误差
fxlist = f(x0);
x_old = x0; % initial point
t = fixt; % step size
for k = 1: itermax  
    % 对于光滑部分做grad梯度下降，对于非光滑部分h使用邻近算子
    x_new = p(x_old - t * grad(x_old),t);
    G = 1/t * (x_old - x_new);      % Gradient mapping  
    fxlist = [fxlist, f(x_new)];
    if norm(G,2)<=tol
        break;
    end
    % 更新
    x_old = x_new;
end
x_end = x_new;
end



