% fista  step size线搜索
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
function [x_end, fxlist]=fista_line_search(f,g,p,grad,x0,itermax,beta)
tol=1e-6;
t0 = 1;   % alpha
x_old=x0;
x_old_old=x0;
fxlist=f(x_old);
for k=1:itermax
    k1=mod(k,1000)+3; % 定期重启
    k1=k;
    y=x_old+(k1-2)/(k1+1)*(x_old-x_old_old);
    direction=grad(y);
    t=t0;
    while true
        x_new=p(y-t*direction,t);
        G = 1/t * (y - x_new);
        if g(x_new) <= g(y) - t*direction'*G + t/2 * norm(G, 2)^2
            break
        else
            t=t*beta;
        end
    end
    G = 1/t * (x_old - x_new);
    fxlist=[fxlist,f(x_new)];

    % 更新
    x_old_old=x_old;
    x_old=x_new;
end
x_end=x_new;
end