# trajectory_generation.py

import os
import numpy as np
import torch
import casadi as ca
from src import *   # from se-hwan/cusadi

# Path to your pre‐generated .casadi function (from Part A)
CASADI_FN_DIR  = "<path_to_where_generate_step.casadi_is_saved>"
CASADI_FN_FILE = "generate_step.casadi"

# -----------------------------
# 1. Choose your gait parameters
# -----------------------------

v_x      = 0.5            # forward velocity (m/s)
v_y      = 0.1            # lateral velocity (m/s)
w_z      = 0.2            # yaw rate (rad/s)
T        = 0.4            # total cycle time (s)
swing_h  = 0.08           # swing foot peak height (m)
N        = 100            # number of discretization steps
nq = 12   # Go2 has 12 joints → nq = 12

# -----------------------------------------------
# 2. Compute ts, p_com = [x_t, y_t, θ_t], in NumPy
# -----------------------------------------------

# 2.1. time vector
ts = np.linspace(0.0, T, N)  # shape (N,)

# 2.2. heading θ(t) = w_z * t
theta_t = w_z * ts          # shape (N,)

# 2.3. x(t), y(t) piecewise (same as your Python code)
if abs(w_z) < 0.01:
    x_t = v_x * ts          # shape (N,)
    y_t = v_y * ts
else:
    x_t = (v_x * np.sin(w_z * ts) + v_y * (np.cos(w_z * ts) - 1.0)) / w_z
    y_t = (v_x * (1.0 - np.cos(w_z * ts)) + v_y * np.sin(w_z * ts)) / w_z

# 2.4. p_com array is not directly needed by generate_step,
#      but we extract x_i, y_i, θ_i per step to feed into CusADi.
#      (phase will be computed separately below.)

# ---------------------------------------------
# 3. Compute “start” and “end” foot positions
# ---------------------------------------------

# We need default_foot_pos in the world frame to build p_foot0, p_foot1.
# We can load q_init (the neutral posture) and compute once with numeric Pinocchio.

import pinocchio as pin

# (a) Numeric Pinocchio model to get default_foot_pos in world frame
model_pin = pin.buildModelFromUrdf("Go2_pinocchio/go2_original.urdf")
data_pin  = model_pin.createData()

q_init_numpy = np.deg2rad(np.array([
    -5.7,  45.8, -86.0,
    +5.7,  45.8, -86.0,
    -5.7,  57.3, -86.0,
    +5.7,  57.3, -86.0
]))  # same neutral posture as in go2_reference_pin.py

pin.forwardKinematics(model_pin, data_pin, q_init_numpy)
pin.updateFramePlacements(model_pin, data_pin)

frame_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
frame_ids   = [model_pin.getFrameId(name) for name in frame_names]

# default_foot_pos_numpy: (4×3) array of foot positions in **world frame** at t=0
default_foot_pos_numpy = np.vstack([
    data_pin.oMf[fid].translation.copy() for fid in frame_ids
])  # shape (4,3)

# (b) Build p_foot0 ∈ ℝ^{4×3} 
#     p_com[0] = (x_t[0], y_t[0], θ_t[0]) (this is t=0)
#     p_foot0 = p_com[0, :2] + R(θ_t[0]) @ default_foot_pos
theta0   = theta_t[0]
c0       = np.cos(theta0)
s0       = np.sin(theta0)
R0       = np.array([[ c0, -s0, 0 ],
                     [ s0,  c0, 0 ],
                     [  0,   0, 1 ]])
p_foot0  = np.zeros((4, 3))
for k in range(4):
    p_world = R0.dot(default_foot_pos_numpy[k])  # (3,)
    p_foot0[k, :] = np.hstack(( x_t[0], y_t[0], 0.0 )) + p_world

# (c) Build p_foot1 ∈ ℝ^{4×3} at t = T (index N-1)
theta_end = theta_t[-1]
ce        = np.cos(theta_end)
se        = np.sin(theta_end)
Re        = np.array([[ ce, -se, 0 ],
                      [ se,  ce, 0 ],
                      [  0,   0, 1 ]])
p_foot1   = np.zeros((4, 3))
for k in range(4):
    p_world_e     = Re.dot(default_foot_pos_numpy[k])  # (3,)
    p_foot1[k, :] = np.hstack(( x_t[-1], y_t[-1], 0.0 )) + p_world_e

# -------------------------------------------------------
# 4. Build per‐step inputs (phase, x_i, y_i, θ_i) for i=0..N-1
# -------------------------------------------------------

phase_t = np.clip((ts / T - 0.25) * 2.0, 0.0, 1.0)  # shape (N,)

# x_t, y_t, θ_t already computed above as (N,)
# Now we will broadcast everything to GPU via PyTorch for CusADi:

# Convert everything to PyTorch CUDA tensors (dtype=double)
device = "cuda"
phase_gpu  = torch.from_numpy(phase_t.astype(np.float64)).to(device)     # (N,)
x_gpu      = torch.from_numpy(x_t.astype(np.float64)).to(device)         # (N,)
y_gpu      = torch.from_numpy(y_t.astype(np.float64)).to(device)         # (N,)
theta_gpu  = torch.from_numpy(theta_t.astype(np.float64)).to(device)     # (N,)

# Broadcast p_foot0, p_foot1 (they are constant across steps) to shape (N,4,3)
p0_tensor = (
    torch.from_numpy(p_foot0.astype(np.float64))
    .reshape(1, 4, 3)
    .repeat(N, 1, 1)
    .to(device)
)
p1_tensor = (
    torch.from_numpy(p_foot1.astype(np.float64))
    .reshape(1, 4, 3)
    .repeat(N, 1, 1)
    .to(device)
)

# Broadcast swing_h, q_init (also constant) to shape (N,)
swing_h_gpu = swing_h * torch.ones((N,), device=device, dtype=torch.double)

q_init_gpu = (
    torch.from_numpy(q_init_numpy.astype(np.float64))
    .reshape(1, nq)
    .repeat(N, 1)
    .to(device)
)  # shape (N,12)

# ------------------------------------------
# 5. Load generate_step.casadi & wrap in CusADi
# ------------------------------------------
fn_casadi_step = ca.Function.load(
    os.path.join(CUSADI_FUNCTION_DIR, "generate_step.casadi")
)

# Create a CusadiFunction that can handle batch‐inputs of size N
fn_cusadi_step = CusadiFunction(fn_casadi_step, N)

# ------------------------------------------
# 6. Evaluate the entire batch (100 steps) in one go
# ------------------------------------------

# Inputs to fn_cusadi_step.evaluate(...) must be a list of length 8, each entry either:
#   - a PyTorch tensor of shape (N,) for scalars, or
#   - a PyTorch tensor of shape (N,4,3) for the 4×3 matrices,
#   - a PyTorch tensor of shape (N,12) for q_init.
#
# Order must match [phase, x_i, y_i, theta, p_foot0, p_foot1, swing_h, q_init].
inputs = [
    phase_gpu,          # (N,)
    x_gpu,              # (N,)
    y_gpu,              # (N,)
    theta_gpu,          # (N,)
    p0_tensor,          # (N,4,3)
    p1_tensor,          # (N,4,3)
    swing_h_gpu,        # (N,)
    q_init_gpu          # (N,12)
]

# Launch GPU kernel (CusADi) to get batch outputs
fn_cusadi_step.evaluate(inputs)

# Retrieve outputs (each is a PyTorch tensor on GPU):
q_out_gpu      = fn_cusadi_step.outputs_sparse[0]  # shape: (N, 12)
foot_flat_gpu  = fn_cusadi_step.outputs_sparse[1]  # shape: (N, 12)

# Synchronize and time (optional)
torch.cuda.synchronize()
# … you could measure GPU time if desired …

# ------------------------------------------
# 7. Convert back to NumPy (CPU) if you want
# ------------------------------------------

q_ref_np       = q_out_gpu.cpu().numpy()        # (100, 12)
foot_flat_np   = foot_flat_gpu.cpu().numpy()    # (100, 12)

# Reshape foot_flat_np → (100, 4, 3)
foot_ref_np    = foot_flat_np.reshape((N, 4, 3))

# Now you have:
#   ts        (N,)
#   q_ref_np  (N,12)
#   foot_ref_np  (N,4,3)
#
# You can save these or plot them as needed.

print("✅ Generated full trajectory for (v_x,v_y,w_z) = "
      f"({v_x:.2f}, {v_y:.2f}, {w_z:.2f})")
print("   ts shape:       ", ts.shape)
print("   q_ref_np shape: ", q_ref_np.shape)
print("   foot_ref_np shape:", foot_ref_np.shape)
