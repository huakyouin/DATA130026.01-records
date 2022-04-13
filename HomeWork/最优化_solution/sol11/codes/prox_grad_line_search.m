function [err, x_end, fxlist] = prox_grad_line_search(A, b, x0, xs, tau, max_iter)
% proximal gradient with with line search for LASSO

beta = 0.5;
t0 = 1;
tol = 1e-6;
f = @(x) 1/2*norm(A*x - b, 2)^2 + tau*norm(x, 1); % objective function
g = @(x) 1/2*norm(A*x - b, 2)^2; % differentiable part
grad = @(x) A'*(A*x - b); % gradient of g
h = @(x) tau*norm(x, 1); % non-differentiable part

fxlist = f(x0);
x_old = x0; % initial point
for k = 1: max_iter
    
    t = t0;
    while true
        x_new = prox(t, tau, x_old - t * grad(x_old));
        G = 1/t * (x_old - x_new);
        if g(x_new) <= g(x_old) - t*grad(x_old)'*G + t/2 * norm(G, 2)^2
            break;
        else
            t = t * beta;
        end
    end

    fxlist = [fxlist, f(x_new)];
   
    if norm(G,2)<=tol
        break;
    end
    
    x_old = x_new;
    
end

x_end = x_new;
err = norm(x_end - xs, 2);