%% 3-2
clear all;clc;close all;
% 生成问题
global A b c m n;
m=500;n=1000;
A=randn(n,m);b=sign(rand(m,1)-0.5);c=0;
fg2 = figure('numbertitle','off','name','第3题（b）');
subplot(121)
semilogy(newton_backtrace(@f32,@J32,@H32,ones(n,1),1e-4,0.5,0.5,'v'));title('牛顿法');ylabel('||JF||_2');xlabel('iters')
subplot(122)
semilogy(GD_backtrace(@f32,@J32,ones(n,1),1e-4,0.5,0.5,'v'));title('梯度法');ylabel('||JF||_2');xlabel('iters')
sgtitle('3-2')
%% 生成题目3-2的函数
function outer=f32(x)
    global A b c m ;
    outer=0.01*(x.'*x+c^2);
    for i = 1:m
        outer = outer+1/m*log(1 + exp(-b(i)*(x'*A(:,i))));
    end
end
function gradient=J32(x)
    global  A b m n;
    gradient=zeros(n,1);
    t=exp(-(A'*x).*b);
    for j=1:n
        for i=1:m
            gradient(j)=gradient(j)-b(i)*A(j,i)*t(i)/(1+t(i));
        end
        gradient(j)=gradient(j)/m+0.02*x(j);
    end
end
function Hessian=H32(x)
    global A b n m;
    Hessian=zeros(n);
    t=exp(-(A'*x).*b);
    t=(t.^2)./((1+t).^2).*(b.^2);
    for j=1:n
        t2=A(j,:)'.*t;
        for k=1:j
            Hessian(j,k)=A(k,:)*t2;
            Hessian(j,k)=Hessian(j,k)/m;
            if j==k
                Hessian(j,k)=Hessian(j,k)+0.02;
            else
                Hessian(k,j)=Hessian(j,k);
            end
        end
    end
end