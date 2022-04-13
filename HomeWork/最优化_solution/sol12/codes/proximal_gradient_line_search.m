function [x, fun_val, gm_val] = proximal_gradient_line_search(f, g, dg, ...
    prox, x0, t0, beta, epsilon, max_iteration)
% INPUT
% ===========================
% f ........ objective function
% dg ....... gradient of the smooth part
% prox ..... proximal operator
% x0 ....... initial point 
% t0 ........ initial stepsize
% beta ..... the constant in which the stepsize is multiplied
%            at each backtracking step (0<beta<1)
% epsilon .. tolerance parameter
% max_iteration .. the maximal iteration number
% OUTPUT
% ============================
% x ........ optimal solution (up to a tolerance)
%            of min f(x)
% fun_val .. record of function value
% gm_val ... record of norm of gradient mapping

gm = @(x, t)(x - prox(x - t*dg(x), t))/t;

t = t0;
x = x0;
iter = 0;
fun_val_new = f(x);
fun_val = [fun_val_new];
gm_new = gm(x, t);
gm_norm = norm(gm_new);
gm_val = [gm_norm];
while (gm_norm >= epsilon && iter < 1000)
    iter = iter+1;
    t = t0;
    gm_new = gm(x, t);
    gm_norm = norm(gm_new);
    while g(x-t*gm_new) > g(x) - t*(dg(x)'*gm_new) + 0.5*t*gm_norm^2
        t = beta * t;
        gm_new = gm(x, t);
        gm_norm = norm(gm_new);
    end
    x = x - t*gm_new;
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
