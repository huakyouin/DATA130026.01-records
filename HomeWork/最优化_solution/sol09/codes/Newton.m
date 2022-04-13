function [funclist, gradlist, solution] = Newton(func, grad, hess, x0, tol)
% Damped Newton's method with Armijo rule line search
%
% Input
% - func: the objective function
% - grad: the gradient of objective function
% - hess: the Hessian of objective function
% - x0: the initial point
% - tol: the convergence tolerance (algorithm terminate when the Euclidean
%        norm of the gradent is smaller than tol)
% Output
% - funclist: the function value of each iteration
% - gradlist: the Euclidean norm of the gradient of each iteration
% - solution: the optimal solution solved by the algorithm

alpha = 0.1;            % parameter of Armijo rule
beta = 0.5;             % parameter of Armijo rule
itermax = 100;          % max iteration number

funclist = zeros(1,itermax);
gradlist = zeros(1,itermax);

% the first iteration
x = x0;
funclist(1) = func(x);
gradlist(1) = norm(grad(x)); 

for iter = 2:itermax
    
    p = -hess(x)\grad(x);   % descent direction, where "\" represents left division
    
    % backtracking line search (Armijo rule)
    t = 1;
    while func(x+t*p) > func(x) + alpha*t*p'*grad(x)
        t = t * beta;
    end
    
    % update
    x = x + t*p;
    funclist(iter) = func(x);
    gradlist(iter) = norm(grad(x));
    
    % stopping criterion
    if norm(grad(x)) <= tol
        break
    end
    
end

solution = x;

end