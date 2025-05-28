import casadi as cs

n = 3
m = 2

H = cs.SX.sym('H', n, n)
g = cs.SX.sym('g', n, 1)
A = cs.SX.sym('A', m, n)
b = cs.SX.sym('b', m, 1)

# Unconstrained solution
x_uncon  = cs.solve(H, -g)          # (n×1)
s_uncon  = b - A@x_uncon            # (m×1)

# Active‐constraint KKT
zero     = cs.SX.zeros(m, m)
KKT      = cs.vertcat(cs.horzcat(H, A.T), cs.horzcat(A, zero))
rhs      = cs.vertcat(-g, b)
sol_act  = cs.solve(KKT, rhs)
x_active = sol_act[0:n]             # (n×1)
lam_act  = sol_act[n:]              # (m×1)

# Build a scalar condition: all slack >= 0 ?
cond = cs.mmin(s_uncon) >= 0        # scalar

# Select branches with a scalar test
x_opt   = cs.if_else(cond, x_uncon,  x_active)   # both (n×1)
lam_opt = cs.if_else(cond,
                    cs.SX.zeros(m,1),             # zero-multiplier branch
                    lam_act)                       # both (m×1)

fn = cs.Function('fn_qp_solver_ineq',
                 [H, g, A, b],
                 [x_opt, lam_opt])
fn.save('fn_qp_solver_ineq_3_2.casadi')
print("Saved fn_qp_solver_ineq_3_2.casadi")
