function [funclist, gradlist, solution] = SGD(func, grad, x0, itermax, rule, para)
% Subgradient method (with three different step size rules)
%
% Input
% - func: the objective function
% - grad: the subgradient of objective function
% - x0: the initial point
% - itermax: the maximum iteration number
% - rule: the step size rule
%         1 : constant step size 
%         2 : constant step length
%         3 : diminishing step size
% - para: the constants or parameters for different step size rules
%         for constant step size, the step size is para
%         for constant step length, the step size is para/norm(gradient)
%         for diminishing step size, the step size is para/sqrt(iter)
% Output
% - funclist: the function value of each iteration
% - gradlist: the Euclidean norm of the gradient of each iteration
% - solution: the optimal solution solved by the algorithm

funclist = zeros(1,itermax);
gradlist = zeros(1,itermax);

% the first iteration
x = x0;
funclist(1) = func(x);
gradlist(1) = norm(grad(x));

for iter = 2:itermax
    
    p = -grad(x);
    
    % step size rule
    if rule == 1            % constant step size
        t = para;
    elseif rule == 2        % constant step length
        t = para/norm(p);
    elseif rule == 3        % diminishing step size
        t = para/sqrt(iter);
    end
    
    % update
    x = x + t*p;
    funclist(iter) = func(x);
    gradlist(iter) = norm(grad(x));
    
end

solution = x;

end

