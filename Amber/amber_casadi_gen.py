#!/usr/bin/env python3
"""
Builds a CasADi-compiled function  “reference_step”  for the Amber biped and
writes it to  amber_reference_step.casadi
Exactly mirrors go2_casadi_gen.py, but for:
  - 2 legs   → 6-D foot vector (2×3)
  - 4 joints → 4-D q vector (q1_L, q2_L, q1_R, q2_R)
"""
import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin
import numpy as np
import os

# ---------------------------------------------------------------------
# 1) User-specified constants (same semantics as Go2 script)
# ---------------------------------------------------------------------
URDF         = "Amber/amber.urdf"             # <-- path to your URDF
FOOT_FRAMES  = ["left_toe", "right_toe"]      # end-effectors in the URDF
DAMPING      = 1e-4
IK_ITERS     = 20
IK_TOL       = 1e-6
N_LEGS       = len(FOOT_FRAMES)               # 2

# ---------------------------------------------------------------------
# 2) Numeric Pinocchio model – only to get frame-IDs & a neutral q-guess
# ---------------------------------------------------------------------
model_num  = pin.buildModelFromUrdf(URDF)
data_num   = model_num.createData()
frame_ids  = [model_num.getFrameId(f) for f in FOOT_FRAMES]
DOF          = model_num.nq                  # 4

q_init_np  = np.zeros(DOF)                    # user asked for “0 for now”

# ---------------------------------------------------------------------
# 3) CasADi-compatible Pinocchio model
# ---------------------------------------------------------------------
cmodel = cpin.Model(model_num)
cdata  = cmodel.createData()

# ---------------------------------------------------------------------
# 4) Declare SX symbols (sizes adapted to Amber)
# ---------------------------------------------------------------------
phase   = ca.SX.sym("phase")
foot_w  = ca.SX.sym("foot_w", N_LEGS*3)       # 6×1
x_com   = ca.SX.sym("x_com")
y_com   = ca.SX.sym("y_com")
theta   = ca.SX.sym("theta")
z_swing = ca.SX.sym("z_swing")
q_cur   = ca.SX.sym("q_cur", DOF)             # 4×1

# 4.1) Reshape & rotations
foot_pos_world = ca.reshape(foot_w, N_LEGS, 3)

c_, s_ = ca.cos(theta), ca.sin(theta)
R_body = ca.vertcat(
    ca.horzcat( c_,  s_, 0),
    ca.horzcat(-s_,  c_, 0),
    ca.horzcat( 0 ,  0 , 1)
)

# 4.2) Desired foot positions in body-frame
foot_pos_body = ca.SX.zeros(N_LEGS, 3)
for k in range(N_LEGS):
    rel = foot_pos_world[k,:].T - ca.vertcat(x_com, y_com, 0)
    pb  = R_body @ rel + ca.vertcat(0, 0, z_swing)
    foot_pos_body[k,:] = pb.T
foot_body_flat = ca.reshape(foot_pos_body, N_LEGS*3, 1)

# ---------------------------------------------------------------------
# 5) Full IK loop in SX (identical logic, smaller matrices)
# ---------------------------------------------------------------------
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
        Jpos = J6[:3, :]                          # 3×4
        Jdls = Jpos.T @ ca.inv(Jpos@Jpos.T + DAMPING*ca.SX.eye(3))
        dq   = Jdls @ err
        q    = ca.if_else(nerr < IK_TOL, q, q + dq)

q_ref = q

# ---------------------------------------------------------------------
# 6) Export CasADi Function  (same signature as Go2 but reduced sizes)
# ---------------------------------------------------------------------
F = ca.Function(
    "reference_step",
    [phase, foot_w, x_com, y_com, theta, z_swing, q_cur],
    [q_ref, foot_body_flat]
)

out_name = "amber_reference_step.casadi"
F.save(out_name)
print(f"[+] wrote {out_name}  ({os.path.getsize(out_name)/1e6:.2f} MB)")
