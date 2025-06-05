#!/usr/bin/env python3
import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin
import numpy as np
import os
print(ca.__version__)
# ---------------------------------------------------------------------
# 1) User‐specified constants (exactly as in your “main” IK wrapper)
# ---------------------------------------------------------------------
URDF = "Go2_pinocchio/go2_original.urdf"
FOOT_FRAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
DAMPING = 1e-4
IK_ITERS = 20      # same as your `max_iter` in the Python loop
IK_TOL = 1e-6

# ---------------------------------------------------------------------
# 2) Build the numeric Pinocchio model + data (only to extract frame IDs)
# ---------------------------------------------------------------------
model_num = pin.buildModelFromUrdf(URDF)
data_num  = model_num.createData()
frame_ids = [model_num.getFrameId(name) for name in FOOT_FRAMES]

# Build a “neutral” initial‐guess q_init (NumPy) exactly as before:
q_init_np = pin.neutral(model_num)
q_init_np[:] = np.deg2rad([
    -5.7, 45.8, -86.0,
    +5.7, 45.8, -86.0,
    -5.7, 57.3, -86.0,
    +5.7, 57.3, -86.0
])

# ---------------------------------------------------------------------
# 3) Build the CasADi (SX)‐compatible Pinocchio model + data
# ---------------------------------------------------------------------
cmodel = cpin.Model(model_num)
cdata  = cmodel.createData()

# ---------------------------------------------------------------------
# 4) Declare SX symbols for inputs of “reference_step”
# ---------------------------------------------------------------------
phase   = ca.SX.sym("phase")        # not used in IK, but part of step inputs
foot_w  = ca.SX.sym("foot_w", 12)   # flattened world‐frame foot positions (4×3)
x_com   = ca.SX.sym("x_com")        # CoM x
y_com   = ca.SX.sym("y_com")        # CoM y
theta   = ca.SX.sym("theta")        # yaw angle
z_swing = ca.SX.sym("z_swing")      # vertical swing height at this phase
q_cur   = ca.SX.sym("q_cur", 12)    # initial guess for IK (12×1)

# Reshape foot_w → (4×3) SX matrix
foot_pos_world = ca.reshape(foot_w, 4, 3)

# ---------------------------------------------------------------------
# 5) Build the 3×3 “world→body” rotation from yaw = theta
# ---------------------------------------------------------------------
c_ = ca.cos(theta)
s_ = ca.sin(theta)
R_body = ca.vertcat(
    ca.horzcat(  c_,  s_, ca.SX(0)),
    ca.horzcat(-s_,  c_, ca.SX(0)),
    ca.horzcat(ca.SX(0), ca.SX(0), ca.SX(1))
)  # SX(3×3)

# ---------------------------------------------------------------------
# 6) Compute desired “foot_pos_body” = R_body*(foot_pos_world−[x;y;0]) + [0;0;z_swing]
# ---------------------------------------------------------------------
foot_pos_body = ca.SX.zeros(4, 3)
for f in range(4):
    pw   = foot_pos_world[f, :].T                 # SX(3×1)
    com0 = ca.vertcat(x_com, y_com, ca.SX(0))      # SX(3×1)
    rel  = pw - com0                              # SX(3×1)
    pb   = R_body @ rel                            # SX(3×1)
    pb   = pb + ca.vertcat(ca.SX(0), ca.SX(0), z_swing)  # add vertical swing
    foot_pos_body[f, :] = pb.T                    # store as (1×3)

# Flatten “foot_pos_body” to 12×1 SX
foot_body_flat = ca.reshape(foot_pos_body, 12, 1)

# ---------------------------------------------------------------------
# 7) IK Logic (nested loops) inside SX
#    Replicates exactly:
#      if q_guess is None: q = q_init.copy() else: q = q_guess.copy()
#      for k,fid in enumerate(foot_frame_ids):
#        for _ in range(max_iter):
#          pin.forwardKinematics(...)
#          pin.updateFramePlacements(...)
#          p_cur = data.oMf[fid].translation
#          err = tgt - p_cur
#          if norm(err)<tol: break
#          J6 = pin.computeFrameJacobian(...)
#          J_pos = J6[:3,:]
#          J_dls = (J_pos^T @ inv(J_pos J_pos^T + λI))   # DLS
#          q += J_dls @ err
#        if norm(err)>=tol: warn(...)
#    In SX, we cannot literally “break” out of an SX‐loop.  Instead, at each iteration
#    we check ‖err‖<tol; if so, we set q_new = q_old, otherwise q_new = q_old + J_dls@err.
# ---------------------------------------------------------------------
q = q_cur  # SX(12×1), to be updated in‐place

for k, fid in enumerate(frame_ids):
    # tgt_k is the (3×1) SX column for foot k
    tgt_k = foot_pos_body[k, :].T  # SX(3×1)

    for it in range(IK_ITERS):
        # 7.1) Forward kinematics & update placements at q
        cpin.forwardKinematics(cmodel, cdata, q)
        cpin.updateFramePlacements(cmodel, cdata)

        # 7.2) Compute current position p_cur_k (3×1 SX)
        p_cur_k = cdata.oMf[fid].translation

        # 7.3) Compute error err_k = tgt - p_cur
        err_k = tgt_k - p_cur_k  # SX(3×1)

        # 7.4) ‖err_k‖₂
        norm_err = ca.norm_2(err_k)  # SX scalar

        # 7.5) Build masked update: if norm_err < IK_TOL, then q stays,
        #      else compute J_pos, J_dls, and do Δq = J_dls @ err_k
        J6_k = cpin.computeFrameJacobian(
            cmodel, cdata, q, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )  # SX(6×12)
        J_pos_k = J6_k[0:3, :]  # SX(3×12)
        λI = DAMPING * ca.SX.eye(3)                             # SX(3×3) damping matrix
        J_dls_k = J_pos_k.T @ ca.inv(J_pos_k @ J_pos_k.T + λI)  # SX(12×3)
        dq_k    = J_dls_k @ err_k   
        # Damped‐Least‐Squares: 12×3 = J_pos_kᵀ * inv(J_pos_k J_pos_kᵀ + λ I₃)
        # J_dls_k = (ca.DM(J_pos_k).T @ ca.inv(ca.DM(J_pos_k) @ ca.DM(J_pos_k).T
        #                      + DAMPING * ca.DM.eye(3))).full()
        dq_k = J_dls_k @ err_k  # SX(12×1)

        # Masked update: if ‖err‖ < tol → q_new = q, else q_new = q + dq_k
        q = ca.if_else(norm_err < IK_TOL, q, q + dq_k)

    # (No warning inside SX; could check externally if needed.)

# After all 4 feet, q holds the final 12×1 SX joint angles
q_ref = q

# ---------------------------------------------------------------------
# 8) Build the CasADi Function “reference_step”
#    Inputs:  [phase, foot_w (12×1), x_com, y_com, theta, z_swing, q_cur]
#    Outputs: [q_ref (12×1), foot_body_flat (12×1)]
# ---------------------------------------------------------------------
F = ca.Function(
    "reference_step",
    [phase, foot_w, x_com, y_com, theta, z_swing, q_cur],
    [q_ref, foot_body_flat],
)

# ---------------------------------------------------------------------
# 9) Save to disk
# ---------------------------------------------------------------------
out_name = "reference_step.casadi"
F.save(out_name)
full_path = os.path.abspath(out_name)
print(f"[+] Wrote “{out_name}” at {full_path} ({os.path.getsize(full_path)/1e6:.2f} MB)")