#!/usr/bin/env python3
"""
Unitree Go2 – full-leg inverse kinematics sanity-check
------------------------------------------------------
*   Calculates joint angles from desired foot positions (IK).
*   Runs forward kinematics with those angles.
*   Prints target vs. achieved positions and error for every foot.
"""

import numpy as np
import pinocchio as pin
from scipy.linalg import pinv

print("Pinocchio version:", pin.__version__)

# ----------------------------------------------------------------------
# Model & constants
# ----------------------------------------------------------------------
GO2_URDF = "Go2_pinocchio/go2_corrected_fixed3.urdf"          # ← path to your converted URDF
FOOT_FRAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
DAMPING      = 1e-4                          # damped-LS regulariser
IK_ITERS     = 100                           # per-foot iterations
IK_TOL       = 1e-6                          # IK stop threshold (m)

model = pin.buildModelFromUrdf(GO2_URDF)
data  = model.createData()

# “Neutral” standing posture as a reasonable initial guess
q_init = pin.neutral(model)
q_init[:] = np.deg2rad(
    [+5.7, 45.8, -86.,
     +5.7, 45.8, -86.,
     -5.7, 57.3, -86.,
     +5.7, 57.3, -86.]
)

frame_ids = [model.getFrameId(name) for name in FOOT_FRAMES]

# ----------------------------------------------------------------------
# IK solver (damped least squares, solved foot-by-foot)
# ----------------------------------------------------------------------
def inverse_kinematics_pinocchio(
    targets_xyz: np.ndarray,
    foot_ids,
    model,
    data,
    q_guess=None,
    tol=1e-6,
    max_iter=100,
    damping=1e-4,
):
    """Multi-foot IK; solves each foot in sequence."""
    if q_guess is None:
        q = q_init.copy()
    else:
        q = q_guess.copy()

    for k, fid in enumerate(foot_ids):
        tgt = targets_xyz[k]
        for _ in range(max_iter):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)

            p_cur = data.oMf[fid].translation
            err   = tgt - p_cur
            if np.linalg.norm(err) < tol:
                break

            J6   = pin.computeFrameJacobian(model, data, q, fid,
                                            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            J    = J6[:3]                       # position rows
            J_dls = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(3))

            q += J_dls @ err

        if np.linalg.norm(err) >= tol:
            print(f"[WARN] Foot {FOOT_FRAMES[k]} did not converge "
                  f"(‖err‖ = {np.linalg.norm(err):.3e} m)")

    return q


# ----------------------------------------------------------------------
# Convenience: forward kinematics → foot positions
# ----------------------------------------------------------------------
def foot_positions_from_q(q, foot_ids, model, data):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return np.vstack([data.oMf[fid].translation.copy() for fid in foot_ids])


# ----------------------------------------------------------------------
# Example target positions (metres, parent frame = base)
# ----------------------------------------------------------------------
targets = np.array([
    [ 0.17782152,  0.11044377, 0.12571125],   # FL
    [ 0.17782152, -0.11044377, 0.12571125],   # FR
    [-0.27051568,  0.11137226, 0.13496522],   # RL
    [-0.27051568, -0.11137226, 0.13496522],   # RR
])

# targets= np.array([
#     [ 0.180,  0.110, 0.126],   # FL
#     [ 0.180, -0.110, 0.126],   # FR
#     [-0.271,  0.111, 0.135],   # RL
#     [-0.271, -0.111, 0.135],   # RR
# ]) 
# ----------------------------------------------------------------------
# 1) Inverse kinematics
# ----------------------------------------------------------------------
q_sol = inverse_kinematics_pinocchio(targets, frame_ids, model, data,
                                     q_guess=q_init, tol=IK_TOL,
                                     max_iter=IK_ITERS, damping=DAMPING)

print("\nSolved joint angles [rad] (4 legs × 3 joints):")
print(q_sol.reshape(4, 3))

# ----------------------------------------------------------------------
# 2) Forward kinematics with the solved angles
# ----------------------------------------------------------------------
achieved = foot_positions_from_q(q_sol, frame_ids, model, data)

# ----------------------------------------------------------------------
# 3) Compare target ↔ achieved
# ----------------------------------------------------------------------
errors = np.linalg.norm(targets - achieved, axis=1)
print("\nTarget vs. achieved foot positions and Cartesian error:")
for i, name in enumerate(FOOT_FRAMES):
    print(f" {name:>6}: target {targets[i]}   →  achieved {achieved[i]}   "
          f"err = {errors[i]:.3e} m")

print("\nMax position error  :", errors.max(), "m")
print("Average position err:", errors.mean(), "m")

if (errors < IK_TOL).all():
    print("\n✅  IK solution validated: all feet within tolerance.")
else:
    print("\n⚠️  IK solution exceeds tolerance on at least one foot.")
