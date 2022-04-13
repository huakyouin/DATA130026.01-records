function [funclist, gradlist, solution] = BBGD(update, func, grad, x0, tol)
% BB-step size gradient descent method with Armijo rule line search
%
% Input
% - update: the update rule of step size
%           1 : t = (s'*y)/(y'*y)
%           2 : t = (s'*s)/(s'*y)
% - func: the objective function
% - grad: the gradient of objective function
% - x0: the initial point
% - tol: the convergence tolerance (algorithm terminate when the Euclidean
%        norm of the gradent is smaller than tol)
% Output
% - funclist: the function value of each iteration
% - gradlist: the Euclidean norm of the gradient of each iteration
% - solution: the optimal solution solved by the algorithm

alpha = 0.1;            % parameter of Armijo rule
beta = 0.5;             % parameter of Armijo rule
armijo_max = 30;        % max Armijo iteration number
itermax = 1000;         % max iteration number

funclist = zeros(1,itermax);
gradlist = zeros(1,itermax);

% the first iteration
x = x0;
funclist(1) = func(x);
gradlist(1) =  norm(grad(x));

% the second iteration
iter = 2;
p = -grad(x);           % descent direction
t = 1;
armijo = 0;
while (func(x+t*p) > func(x) + alpha*t*p'*grad(x)) && (armijo < armijo_max)
    t = t * beta;
    armijo = armijo + 1;
end
x_old = x;
x = x + t*p;                        % update x
funclist(iter) = func(x);           % update function value
gradlist(iter) = norm(grad(x));     % update norm of gradient
s = x - x_old;                      % compute s
y = grad(x) - grad(x_old);          % compute y

for iter = 3:itermax
    
    p = -grad(x);               % descent direction
    
    if update == 1
        t = (s'*y)/(y'*y);      % the first update rule
    else
        t = (s'*s)/(s'*y);      % the second update rule
    end
    t = min(abs(t),1e4);        % ensure stability
    
    % Armijo rule
    armijo = 0;
    while (func(x+t*p) > func(x) + alpha*t*p'*grad(x)) && (armijo < armijo_max)
        t = t * beta;
        armijo = armijo + 1;
    end

    % update
    x_old = x;
    x = x + t*p;
    funclist(iter) = func(x);
    gradlist(iter) = norm(grad(x));
    
    % stopping criterion
    if norm(grad(x)) <= tol    
        break
    end
    
    % update
    s = x - x_old;
    y = grad(x) - grad(x_old);
       
end

solution = x;

end