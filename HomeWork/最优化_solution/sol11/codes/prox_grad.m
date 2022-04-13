function [err, x_end, fxlist] = prox_grad(A, b, x0, xs, tau, max_iter)
% proximal gradient with constant step size 1/norm(A'*A) for LASSO

tol = 1e-6;
f = @(x) 1/2*norm(A*x - b, 2)^2 + tau*norm(x, 1); % objective function
g = @(x) 1/2*norm(A*x - b, 2)^2; % differentiable part
grad = @(x) A'*(A*x - b); % gradient of g
h = @(x) tau*norm(x, 1); % non-differentiable part

fxlist = f(x0);
x_old = x0; % initial point
t = 1/norm(A'*A, 2); % step size

for k = 1: max_iter
    
    x_new = prox(t, tau, x_old - t * grad(x_old));
    G = 1/t * (x_old - x_new);      % Gradient mapping
    
    fxlist = [fxlist, f(x_new)];

    if norm(G,2)<=tol
        break;
    end
    
    x_old = x_new;
    
end

x_end = x_new;
err = norm(x_end - xs, 2);



