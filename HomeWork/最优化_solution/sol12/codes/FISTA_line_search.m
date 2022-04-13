function [x_new, fun_val, dg_val] = FISTA_line_search(f, g, dg, prox, x0, ...
    t0, beta, epsilon, max_iteration)
% INPUT
% ===========================
% f ........ objective function
% dg ....... gradient of smooth function
% prox ..... proximal operator
% x0 ....... initial point 
% t ........ constant stepsize
% epsilon .. tolerance parameter
% beta ..... the constant in which the stepsize is multiplied
%            at each backtracking step (0<beta<1)
% max_iteration .. the maximal iteration number
% OUTPUT
% ============================
% x ........ optimal solution (up to a tolerance)
%            of min f(x)
% fun_val .. record of function value

x_new = x0;
x_old = x0;
t = t0;
iter = 0;
fun_val_new = f(x_new);
fun_val = [fun_val_new];
dg_norm = norm(dg(x_new));
dg_val = [dg_norm];
while iter < max_iteration && dg_norm > epsilon
    iter = iter+1;
    y = x_new + (iter-2)/(iter+1)*(x_new-x_old);
    x_old = x_new;
    x = prox(y - t * dg(y),t);
    while g(x) > g(y)+dg(y)'*(x-y)+norm(x-y)^2/(2*t)
        t = beta*t;
        x = prox(y - t*dg(y), t);
    end
    x_new = x;
    fun_val_new = f(x_new);
    fun_val = [fun_val fun_val_new];
    dg_norm = norm(dg(x_new));
    dg_val = [dg_val dg_norm];
    if rem(iter, 200) == 0
        fprintf('iter_number = %3d norm_grad = %2.6f fun_val = %2.6f\n',...
            iter, dg_norm, fun_val_new);
    end
end


if (iter==max_iteration)
    fprintf('Max iteration number!\n')
end
