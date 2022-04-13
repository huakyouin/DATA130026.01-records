function [solution, funclist, gradlist] = gradient_method_constant(func, grad, x0, t, tol, itermax)
% Gradient descent method with constant step size

funclist = zeros(1,itermax);
gradlist = zeros(1,itermax);

% the first iteration
x = x0;
funclist(1) = func(x);
gradlist(1) = norm(grad(x)); 

for iter = 2:itermax
    
    p = -grad(x);               % descent direction
    
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