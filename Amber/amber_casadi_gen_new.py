#!/usr/bin/env python3
import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin
import numpy as np
import os

# ---------------------------------------------------------------------
# 1) User-specified constants (similar to go2 version)
# ---------------------------------------------------------------------
URDF        = "Amber/amber_free.urdf"
FOOT_FRAMES = ["left_toe", "right_toe"]
DAMPING     = 1e-4
IK_ITERS    = 20
IK_TOL      = 1e-6

# ---------------------------------------------------------------------
# 2) Build the numeric Pinocchio model + data (to extract frame IDs & defaults)
# ---------------------------------------------------------------------
model_num = pin.buildModelFromUrdf(URDF)
data_num  = model_num.createData()
frame_ids = [model_num.getFrameId(f) for f in FOOT_FRAMES]

# Extract each foot’s default y-offset and z-stance height
default_foot_pos = [data_num.oMf[fid].translation.copy() for fid in frame_ids]
foot_y_offset    = [p[1] for p in default_foot_pos]
foot_z_stance    = [p[2] for p in default_foot_pos]

# ---------------------------------------------------------------------
# 3) Build the CasADi (SX)-compatible Pinocchio model + data
# ---------------------------------------------------------------------
cmodel = cpin.Model(model_num)
cdata  = cmodel.createData()

# ---------------------------------------------------------------------
# 4) Declare SX symbols for inputs of “amber_reference_step”
# ---------------------------------------------------------------------
phase   = ca.SX.sym("phase")         # just passed through
foot_x  = ca.SX.sym("foot_x", 2)     # only X positions of the 2 feet
x_com   = ca.SX.sym("x_com")         # COM X
z_com   = ca.SX.sym("z_com")         # COM Z (height)
z_swing = ca.SX.sym("z_swing")       # swing height
q_cur   = ca.SX.sym("q_cur", 4)      # current guess for the 4 joint angles

# ---------------------------------------------------------------------
# 5) Reconstruct each foot’s full 3D world target from its X input
# ---------------------------------------------------------------------
# Build a (2×3) SX for [x, y_offset, z_stance]
foot_w = ca.SX.zeros(2,3)
for i in range(2):
    foot_w[i,0] = foot_x[i]
    foot_w[i,1] = foot_y_offset[i]   # numeric constant
    foot_w[i,2] = foot_z_stance[i]    # numeric constant

# ---------------------------------------------------------------------
# 6) Compute foot positions in the *body* frame:
#    rel = foot_w − [x_com; 0; z_com], then add vertical swing
# ---------------------------------------------------------------------
foot_body = ca.SX.zeros(2,3)
for i in range(2):
    pw  = foot_w[i,:].T                             # SX(3×1)
    com = ca.vertcat(x_com, ca.SX(0), z_com)         # SX(3×1)
    rel = pw - com                                  # SX(3×1)
    pb  = rel + ca.vertcat(ca.SX(0), ca.SX(0), z_swing)
    foot_body[i,:] = pb.T

# Flatten to a 6×1 vector
foot_body_flat = ca.reshape(foot_body, 6, 1)

# ---------------------------------------------------------------------
# 7) IK Logic (exactly as in go2: damped‐least‐squares per foot)
# ---------------------------------------------------------------------
# Here we assume the model has exactly 4 actuated joints in the same order as q_cur
q = q_cur  # SX(4×1)

for k, fid in enumerate(frame_ids):
    tgt_k = foot_body[k,:].T  # SX(3×1)
    for _ in range(IK_ITERS):
        # 7.1) FK & frame placements
        cpin.forwardKinematics(cmodel, cdata, q)
        cpin.updateFramePlacements(cmodel, cdata)
        # 7.2) Current foot pos
        p_cur = cdata.oMf[fid].translation       # SX(3×1)
        err   = tgt_k - p_cur                    # SX(3×1)
        norm_err = ca.norm_2(err)                # SX scalar
        # 7.3) Jacobian & DLS
        J6   = cpin.computeFrameJacobian(
            cmodel, cdata, q, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )                                       # SX(6×4)
        Jpos = J6[0:3,:]                        # SX(3×4)
        λI   = DAMPING * ca.SX.eye(3)           # SX(3×3)
        Jdls = Jpos.T @ ca.inv(Jpos @ Jpos.T + λI)  # SX(4×3)
        dq   = Jdls @ err                       # SX(4×1)
        # 7.4) Mask update
        q    = ca.if_else(norm_err < IK_TOL, q, q + dq)

q_ref = q  # final 4×1 SX

# ---------------------------------------------------------------------
# 8) Build & save the CasADi Function
# ---------------------------------------------------------------------
F = ca.Function(
    "amber_reference_step",
    [phase, foot_x, x_com, z_com, z_swing, q_cur],
    [q_ref, foot_body_flat],
)
out_name = "amber_reference_step.casadi"
F.save(out_name)
print(f"[+] Wrote “{out_name}” at {os.path.abspath(out_name)}")
