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
function [x_end, fxlist]=fista(f,p,grad,fixt,x0,itermax)
tol= 1e-6;
x_old=x0;
x_old_old=x0;
fxlist=f(x0);
t = fixt; % step size
for k=1:itermax
    k1=mod(k,1000)+2;% 定期重启
    y=x_old+(k1-2)/(k1+1)*(x_old-x_old_old);
    x_old_old=x_old;
    x_new=p(y-t*grad(y),t);
    fxlist=[fxlist,f(x_new)];
    G = 1/t * (x_old - x_new);      % Gradient mapping  
    if norm(G,2)<= tol
        break;
    end
    x_old_old=x_old;
    x_old=x_new;
end
x_end=x_new;
end