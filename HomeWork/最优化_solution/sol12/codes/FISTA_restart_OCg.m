 function [x_new, fun_val, dg_val] = FISTA_restart_OCg(f, dg, prox, x0, ...
    t, epsilon, max_iteration)
% INPUT
% ===========================
% f ........ objective function
% dg ....... gradient of smooth function
% prox ..... proximal operator
% x0 ....... initial point 
% t ........ constant stepsize
% epsilon .. tolerance parameter
% max_iteration .. the maximal iteration number
% OUTPUT
% ============================
% x ........ optimal solution (up to a tolerance)
%            of min f(x)
% fun_val .. record of function value

x_new = x0;
x_old = x0;
iter = 0;
k = 0;
fun_val_new = f(x_new);
fun_val = [fun_val_new];
dg_norm = norm(dg(x_new));
dg_val = [dg_norm];
while iter < max_iteration && dg_norm > epsilon
    iter = iter+1;
    k = k+1;
    y = x_new + (k-2)/(k+1)*(x_new-x_old);
    x_old = x_new;
    x_new = prox(y - t * dg(y),t);
    fun_val_new = f(x_new);
    fun_val = [fun_val fun_val_new];
    dg_norm = norm(dg(x_new));
    dg_val = [dg_val dg_norm];
    if dg(y)'*(x_new - x_old) > 0
        k = 0;
    end
    if rem(iter, 200) == 0
        fprintf('iter_number = %3d norm_grad = %2.6f fun_val = %2.6f\n',...
            iter, dg_norm, fun_val_new);
    end
end


if (iter==max_iteration)
    fprintf('Max iteration number!\n')
end
