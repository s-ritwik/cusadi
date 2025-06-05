# debug_initial_foot.py

import numpy as np
import pinocchio as pin

# (A) Reproduce exactly how you build default_foot_pos_numpy
GO2_URDF = "Go2_pinocchio/go2_original.urdf"
model_pin = pin.buildModelFromUrdf(GO2_URDF)
data_pin  = model_pin.createData()

# The same neutral joints:
q_init_numpy = np.deg2rad(np.array([
    -5.7,  45.8, -86.0,    # FL hip, thigh, calf
    +5.7,  45.8, -86.0,    # FR hip, thigh, calf
    -5.7,  57.3, -86.0,    # RL hip, thigh, calf
    +5.7,  57.3, -86.0     # RR hip, thigh, calf
]))

# Compute forward kinematics (numeric) to get "default" foot pos
pin.forwardKinematics(model_pin, data_pin, q_init_numpy)
pin.updateFramePlacements(model_pin, data_pin)

FOOT_FRAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
frame_ids = [model_pin.getFrameId(name) for name in FOOT_FRAMES]
default_foot_pos_numpy = np.vstack([
    data_pin.oMf[fid].translation.copy() for fid in frame_ids
])  # shape (4,3)

print("Numeric default_foot_pos_numpy (from Pinocchio):")
for idx, name in enumerate(FOOT_FRAMES):
    print(f"  {name}: {default_foot_pos_numpy[idx]}")

# (B) Build exactly the same R0 you used in your GPU code
theta0 = 0.0  # at t=0, COM heading is zero
c0, s0 = np.cos(theta0), np.sin(theta0)
R0_gpu = np.array([[ c0,  s0, 0],
                   [ -s0, c0, 0],
                   [   0,   0, 1]])  # << your “fixed” version

# Also show the “standard” rotation for comparison:
R0_std = np.array([[c0, -s0, 0],
                   [s0,  c0, 0],
                   [ 0,   0, 1]])

print("\nR0_gpu   (used in your GPU code) =\n", R0_gpu)
print("R0_std   (the usual [c -s; s c] form)  =\n", R0_std)

# (C) Compute the “target” foot positions at t=0 that your CasADi function sees:
#     base_pos0 = [x(0), y(0), 0] = [0,0,0] since ts[0]=0 → x=0, y=0
base_pos0 = np.array([0.0, 0.0, 0.0])
target_foot_gpu = np.zeros((4,3))
for k in range(4):
    target_foot_gpu[k] = base_pos0 + R0_gpu.dot(default_foot_pos_numpy[k])

print("\nTarget foot positions at t=0  (from GPU‐code R0 × default):")
for idx, name in enumerate(FOOT_FRAMES):
    print(f"  {name}: {target_foot_gpu[idx]}")

# (D) Now compute “current” foot positions from Pinocchio with q_init again,
#     but remember: data_pin.oMf[fid].translation is already in body coords
print("\nCurrent foot positions from numeric FK (same as default_foot_pos_numpy):")
for idx, name in enumerate(FOOT_FRAMES):
    print(f"  {name}: {data_pin.oMf[frame_ids[idx]].translation.copy()}")

# (E) Compare elementwise:
print("\nDifference (target_gpu - current_pin) at t=0:")
for idx, name in enumerate(FOOT_FRAMES):
    diff = target_foot_gpu[idx] - data_pin.oMf[frame_ids[idx]].translation.copy()
    print(f"  {name}: {diff}   (norm = {np.linalg.norm(diff):.3e})")
