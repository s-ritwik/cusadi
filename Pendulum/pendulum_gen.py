import casadi as ca

# Symbolic expressions
x_pend = ca.SX.sym('x_pend', 2, 1)    # pendulum state [θ; ω]
g      = ca.SX.sym('g',      1, 1)    # gravity
l      = ca.SX.sym('l',      1, 1)    # pendulum length
dt     = ca.SX.sym('dt',     1, 1)    # timestep

# Continuous‐time dynamics
f_pend = ca.vertcat(
    x_pend[1],
    -g * ca.sin(x_pend[0]) / l
)

# Jacobian w.r.t. state
J_pend = ca.jacobian(f_pend, x_pend)

# Semi‐implicit Euler step
omega_next   = x_pend[1] - (g * ca.sin(x_pend[0]) / l) * dt
theta_next   = x_pend[0] + omega_next * dt
x_next_pend  = ca.vertcat(theta_next, omega_next)

# Wrap as CasADi Functions
fn_dynamics = ca.Function('fn_dynamics',
                          [x_pend, g, l],
                          [f_pend])

fn_jacobian = ca.Function('fn_jacobian',
                          [x_pend, g, l],
                          [J_pend])

fn_sim_step = ca.Function('fn_sim_step',
                          [x_pend, g, l, dt],
                          [x_next_pend])

# Save to .casadi files
fn_dynamics.save('fn_dynamics.casadi')
fn_sim_step.save('fn_sim_step.casadi')
fn_jacobian.save('fn_jacobian.casadi')
