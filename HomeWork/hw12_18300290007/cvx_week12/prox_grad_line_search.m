% 投影梯度法  step size线搜索
%----------输入参数
% - f: 原函数
% - g:函数可微部分
% - p：函数不可微部分的正交投影函数
% - grad：函数可微部分梯度
% - x0: 迭代起点
% - itermax: 最大迭代步数
% - beta:回溯倍率
%----------
%----------输出说明
% - fxlist: 每步迭代函数值列表
% - x_end: 算法给出最优解（未必真最优）
%----------
function [x_end, fxlist] = prox_grad_line_search(f,g,p,grad,x0,itermax,beta)
t0 = 1;   % alpha
tol = 1e-6; % 容忍误差
fxlist = f(x0);
x_old = x0; % 起始点
for k = 1: itermax
    t = t0;
    direction=grad(x_old);
    while true
        x_new = p(x_old - t * direction,t);
        G = 1/t * (x_old - x_new);
        if g(x_new) <= g(x_old) - t*direction'*G + t/2 * norm(G, 2)^2
            break;
        else
            t = t * beta;
        end
    end
    fxlist = [fxlist, f(x_new)];
    % 更新
    x_old = x_new;
end
x_end = x_new;
end