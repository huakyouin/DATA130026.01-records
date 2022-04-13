% 投影函数
%------- 输入说明
% - t：
function y = prox(t,tau,x)
y  = zeros(length(x),1);
% proximal mapping of tau*norm(x,1)
h1=@(x) tau*norm(x, 1);
if h == h1

    ind = find(x>t*tau);
    y(ind) = x(ind)- t*tau;
    ind = find(x<-t*tau);
    y(ind) = x(ind)+ t*tau;
end
% 没有h时
if h==0 
    y=x;
end

