function [x, fun_val, gm_val] = proximal_gradient_constant_size(f, dg, ...
    prox, x0, t, epsilon, max_iteration)
% INPUT
% ===========================
% f ........ objective function
% dg ....... gradient of the smooth part
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
% gm_val ... record of norm of gradient mapping

gm = @(x, t)(x - prox(x - t*dg(x), t))/t;

x = x0;
iter = 0;
fun_val_new = f(x);
fun_val = [fun_val_new];
gm_new = gm(x, t);
gm_norm = norm(gm_new);
gm_val = [gm_norm];
while (gm_norm >= epsilon && iter < max_iteration)
    iter = iter+1;
    x = x - t*gm_new;
    gm_new = gm(x, t);
    gm_norm = norm(gm_new);
    gm_val = [gm_val gm_norm];
    fun_val_new = f(x);
    fun_val = [fun_val fun_val_new];
    if rem(iter, 200) == 0
        fprintf('iter_number = %3d norm_gm = %2.6f fun_val = %2.6f\n',...
            iter, gm_norm, fun_val_new);
    end
end

if (iter==max_iteration)
    fprintf('Max iteration number!\n')
end
