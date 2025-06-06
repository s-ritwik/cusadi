#!/usr/bin/env python3
import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin

# ----------------------------
# 0.  Constants & URDF
# ----------------------------
URDF        = "Go2_pinocchio/go2_original.urdf"
FOOT_FRAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
DAMPING     = 1e-4
IK_ITERS    = 20
TOL         = 1e-6

# ----------------------------
# 1.  Build numeric & CasADi models
# ----------------------------
model_num = pin.buildModelFromUrdf(URDF)
data_num  = model_num.createData()
frame_ids = [model_num.getFrameId(n) for n in FOOT_FRAMES]

# CasADi‐SX version of the same model:
cmodel = cpin.Model(model_num)
cdata  = cmodel.createData()

nq = model_num.nq  # should be 12

# ----------------------------
# 2.  CasADi symbols
# ----------------------------
phase   = ca.SX.sym("phase")           # ∈ [0,1]
x_i     = ca.SX.sym("x_i")
y_i     = ca.SX.sym("y_i")
theta_i = ca.SX.sym("theta")

p_foot0 = ca.SX.sym("p_foot0", 4, 3)    # world-frame keyframe at t=0
p_foot1 = ca.SX.sym("p_foot1", 4, 3)    # world-frame keyframe at t=T
swing_h = ca.SX.sym("swing_h")

q_init  = ca.SX.sym("q_init", nq)       # initial guess (12×1)

# ----------------------------
# 3.  Swing‐height (symmetric Bézier)
# ----------------------------
u       = phase * 2                      # 0→1 up, 1→2 down
up      = u <= 1
u_up    = u
u_dn    = u - 1

bez = lambda uu: uu**3 + 3*(uu**2)*(1 - uu)

z_up = swing_h * bez(u_up)
z_dn = swing_h - swing_h * bez(u_dn)
z    = ca.if_else(up, z_up, z_dn)        # final swing‐height (scalar SX)

# ----------------------------
# 4.  Foot path in WORLD (cubic Bézier)
# ----------------------------
s_bez     = bez(phase)
foot_world = p_foot0 + (p_foot1 - p_foot0) * s_bez  # (4×3)
# add z to each foot’s z‐coordinate:
fz        = foot_world[:, 2] + z
foot_world = ca.horzcat(foot_world[:, 0], foot_world[:, 1], fz)

# ----------------------------
# 5.  Multi‐foot DLS IK (in WORLD frame)
# ----------------------------
q = q_init                  # SX(12×1)
flags = []

for k, fid in enumerate(frame_ids):
    # Loop up to IK_ITERS, break early if error < TOL:
    for _ in range(IK_ITERS):
        # 5.1) Forward‐kin and Jacobian
        cpin.forwardKinematics(cmodel, cdata, q)
        cpin.computeJointJacobians(cmodel, cdata, q)
        cpin.updateFramePlacements(cmodel, cdata)

        p_cur = cdata.oMf[fid].translation          # SX(3×1)
        err = ca.reshape(foot_world[k, :], (3,1)) - p_cur

        cond = ca.norm_2(err) < TOL                 # SX boolean

        # 5.2) 3×12 Jacobian (world‐aligned)
        J6   = cpin.computeFrameJacobian(
                   cmodel, cdata, q,
                   fid,
                   cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED
               )                                  # SX(6×12)
        J    = J6[0:3, :]                          # SX(3×12)

        A    = J @ J.T + DAMPING * ca.SX_eye(3)    # SX(3×3)
        dq   = J.T @ ca.solve(A, err)              # SX(12×1)

        # 5.3) q ← (cond ? q : q + dq)
        q = ca.if_else(cond, q, q + dq)

    # After IK_ITERS, check final error & record flag:
    cpin.forwardKinematics(cmodel, cdata, q)
    cpin.updateFramePlacements(cmodel, cdata)
    p_fin  = cdata.oMf[fid].translation           # SX(3×1)
    # err_fin = foot_world[k, :] - p_fin             # SX(3×1)
    err_fin = ca.reshape(foot_world[k, :], (3,1)) - p_fin

    flags.append( ca.if_else(ca.norm_2(err_fin) < TOL, 1, 0) )

flags_vec = ca.vertcat(*flags)  # SX(4×1), each entry ∈ {0,1}

# ----------------------------
# 6.  Build body‐frame foot positions (optional logging)
# ----------------------------
c  = ca.cos(theta_i);  s  = ca.sin(theta_i)
Rwb = ca.horzcat(
    ca.vertcat(c, -s, ca.SX(0.0)),
    ca.vertcat(s,  c, ca.SX(0.0)),
    ca.vertcat(ca.SX(0.0), ca.SX(0.0), ca.SX(1.0))
)
com = ca.vertcat(x_i, y_i, 0)                  # (3×1)
delta = foot_world - ca.repmat(com.T, 4, 1)    # (4×3)
foot_body = (Rwb @ delta.T).T                  # (4×3)

foot_flat = ca.reshape(foot_body, 12, 1)       # (12×1)

# ----------------------------
# 7.  Pack into CasADi Function
# ----------------------------
generate_step = ca.Function(
    "generate_step",
    [phase, x_i, y_i, theta_i, p_foot0, p_foot1, swing_h, q_init],
    [q, foot_flat, flags_vec],
    ["phase","x","y","theta","p0","p1","h","q0"],
    ["q_out","foot_flat","flags"]
)

generate_step.save("generate_step.casadi")
print("✅  generate_step.casadi written (world‐frame DLS IK, flags included)")