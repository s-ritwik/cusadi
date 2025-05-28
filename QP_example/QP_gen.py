# qp_gen.py ---------------------------------------------------------------
import casadi as ca

# ---------- problem size (edit freely) ----------
n = 3       # number of decision variables
m = 1     # number of equality constraints  Ax = b
# ------------------------------------------------

# symbolic variables ------------------------------------------------------
H = ca.SX.sym('H', n, n)      # SPD Hessian
g = ca.SX.sym('g', n, 1)      # gradient
A = ca.SX.sym('A', m, n)      # constraint matrix
b = ca.SX.sym('b', m, 1)      # RHS
# ------------------------------------------------------------------------

# KKT system:  [ H  Aᵀ ] [x] = [-g]
#              [ A   0 ] [λ]   [ b]
K   = ca.vertcat(ca.hcat([H, A.T]),
                 ca.hcat([A, ca.SX.zeros(m, m)]))
rhs = ca.vertcat(-g, b)

sol = ca.solve(K, rhs)        # analytic solution
x_opt = sol[0:n]              # optimal decision vars
lam   = sol[n:]               # Lagrange multipliers

fn_qp = ca.Function('fn_qp_solver',
                    [H, g, A, b],
                    [x_opt, lam],
                    ['H', 'g', 'A', 'b'],
                    ['x', 'lambda'])

fn_qp.save('fn_qp_solver_3_1.casadi')
print('Saved  fn_qp_solver.casadi')
