#!/usr/bin/env python3
"""
Builds a CasADi-compiled IK step for the planar Amber,
where each foot target is specified by an x-offset and a common swing height.
"""
import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin
import numpy as np
import os

# ——————————————————————————————
# 1) User constants
# ——————————————————————————————
URDF        = "Amber/amber_free.urdf"
FOOT_FRAMES = ["left_toe", "right_toe"]
DAMPING     = 1e-4
IK_ITERS    = 20
IK_TOL      = 1e-6
N_LEGS      = len(FOOT_FRAMES)    # 2

# ——————————————————————————————
# 2) Numeric Pinocchio model (to get frame IDs)
# ——————————————————————————————
model_num  = pin.buildModelFromUrdf(URDF)
data_num   = model_num.createData()
frame_ids  = [model_num.getFrameId(f) for f in FOOT_FRAMES]
DOF        = model_num.nq        # 4 joint actuators

# ——————————————————————————————
# 3) CasADi Pinocchio model
# ——————————————————————————————
cmodel = cpin.Model(model_num)
cdata  = cmodel.createData()

# ——————————————————————————————
# 4) SX symbols
# ——————————————————————————————
phase    = ca.SX.sym("phase")                # unused but kept for parity
foot_x   = ca.SX.sym("foot_x", N_LEGS)       # 2×1
z_swing  = ca.SX.sym("z_swing")              # common swing height
q_cur    = ca.SX.sym("q_cur", DOF)           # 4×1

# ——————————————————————————————
# 5) Build desired foot pos in body-frame
# ——————————————————————————————
# each foot: ( x_offset, 0, z_swing )
foot_pos_body = ca.SX.zeros(N_LEGS, 3)
for k in range(N_LEGS):
    foot_pos_body[k,0] = foot_x[k]
    foot_pos_body[k,1] = 0
    foot_pos_body[k,2] = z_swing

# flatten for output (6×1)
foot_body_flat = ca.reshape(foot_pos_body, N_LEGS*3, 1)

# ——————————————————————————————
# 6) Damped‐least‐squares IK loop
# ——————————————————————————————
q = q_cur
for k, fid in enumerate(frame_ids):
    tgt = foot_pos_body[k,:].T
    for _ in range(IK_ITERS):
        cpin.forwardKinematics(cmodel, cdata, q)
        cpin.updateFramePlacements(cmodel, cdata)
        err  = tgt - cdata.oMf[fid].translation
        nerr = ca.norm_2(err)
        J6   = cpin.computeFrameJacobian(
                  cmodel, cdata, q, fid,
                  pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        Jpos = J6[:3, :]  
        Jdls = Jpos.T @ ca.inv(Jpos @ Jpos.T + DAMPING*ca.SX.eye(3))
        dq   = Jdls @ err
        # if error small, stop; else step
        q    = ca.if_else(nerr < IK_TOL, q, q + dq)

q_ref = q

# ——————————————————————————————
# 7) Export the CasADi function
# ——————————————————————————————
F = ca.Function(
    "reference_step",
    [ phase,   foot_x,   z_swing,   q_cur ],
    [ q_ref,   foot_body_flat ]
)
out_name = "amber_reference_step.casadi"
F.save(out_name)
print(f"[+] Wrote {out_name} ({os.path.getsize(out_name)/1e6:.2f} MB)")
