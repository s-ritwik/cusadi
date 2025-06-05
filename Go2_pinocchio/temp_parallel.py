# go2_IK_solver.py

import os
import numpy as np
import casadi as ca
import pinocchio as pin

# -------------------------------
# 1) Load Pinocchio numeric model → default foot positions (world frame)
# -------------------------------
GO2_URDF = "Go2_pinocchio/go2_original.urdf"
model_pin = pin.buildModelFromUrdf(GO2_URDF)
data_pin  = model_pin.createData()

# Neutral “standing” joint angles (radians)
q_init = pin.neutral(model_pin)
q_init[:] = np.deg2rad([
    -5.7, 45.8, -86.0,
     +5.7, 45.8, -86.0,
    -5.7, 57.3, -86.0,
     +5.7, 57.3, -86.0
])

# Compute forward kinematics at neutral → default foot positions
pin.forwardKinematics(model_pin, data_pin, q_init)
pin.updateFramePlacements(model_pin, data_pin)
FOOT_FRAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
frame_ids    = [model_pin.getFrameId(name) for name in FOOT_FRAMES]
default_foot_positions = {}
for fid in frame_ids:
    default_foot_positions[fid] = data_pin.oMf[fid].translation.copy()  # NumPy (3,)

# Build a 4×3 NumPy matrix of default foot positions (world)
default_foot_mat = np.vstack([default_foot_positions[fid] for fid in frame_ids])  # shape (4,3)

# -------------------------------
# 2) Velocity grids (same dims as original generate_gait_libray)
# -------------------------------
v_xs = np.linspace(-1.0, 1.5, 11)   # 11 samples
v_ys = np.linspace(-0.75, 0.75, 7)  # 7 samples
w_zs = np.linspace(-0.5, 0.5, 5)    # 5 samples

vx_len = v_xs.size
vy_len = v_ys.size
wz_len = w_zs.size

# Save the grids immediately (to match generate_gait_libray)
np.save("references/go2_vxs.npy", v_xs)
np.save("references/go2_vys.npy", v_ys)
np.save("references/go2_wzs.npy", w_zs)

# -------------------------------
# 3) CasADi – load the single‐step IK function
# -------------------------------
ik_func = ca.Function.load("src/casadi_functions/go2_IK_step.casadi")

# We’ll call it in “parallel” over N time steps via casadi.map(...)
# ==============================================================================
# >>> For each (v_x, v_y, w_z), we build these arrays (all NumPy), then convert
#     to CasADi DM, map the IK function over them in batch.
# ==============================================================================

# Prepare structures to hold per‐combination results
#   q_refs_all.shape    = (vx_len, vy_len, wz_len, N, 12)
#   foot_refs_all.shape = (vx_len, vy_len, wz_len, N, 4, 3)
#   ts_out will be common across all combos (1×N)

# Initialize containers
# We do not yet know N; we’ll fix it once we pick a gait cycle length.
# In the original, N=100 (hard‐coded). We’ll do the same.
N = 100
q_refs_all    = np.zeros((vx_len, vy_len, wz_len, N, 12))
foot_refs_all = np.zeros((vx_len, vy_len, wz_len, N, 4, 3))
ts_out_global = None   # will be set on first iteration

# -------------------------------
# 4) Loop over each (v_x, v_y, w_z) combination
# -------------------------------
for ix, vx in enumerate(v_xs):
    for iy, vy in enumerate(v_ys):
        for iz, wz in enumerate(w_zs):
            print(f"Generating reference for (v_x, v_y, w_z) = ({vx:.2f}, {vy:.2f}, {wz:.2f})")

            # 4.1) Build time vector ts (NumPy) and COM motion
            T = 0.4                     # total gait cycle duration
            ts = np.linspace(0.0, T, N) # shape (N,)
            if ts_out_global is None:
                ts_out_global = ts.copy()
                np.save("references/go2_reference_ts.npy", ts_out_global)

            # Heading over time
            theta = wz * ts            # shape (N,)

            # COM x,y over time (NumPy)
            if abs(wz) < 1e-2:
                x_t = vx * ts
                y_t = vy * ts
            else:
                x_t = (vx * np.sin(wz*ts) + vy * (np.cos(wz*ts) - 1.0)) / wz
                y_t = (vx * (1.0 - np.cos(wz*ts)) + vy * np.sin(wz*ts)) / wz

            # 4.2) Build phase array and swing height “z_i” array (NumPy)
            phase = np.clip((ts / T - 0.25) * 2.0, 0.0, 1.0)  # shape (N,)
            z_all = np.zeros(N)
            for i in range(N):
                if phase[i] <= 0.5:
                    t_norm = 2 * phase[i]
                    bez = t_norm**3 + 3*(t_norm**2)*(1 - t_norm)
                    z_all[i] = 0.08 * bez
                else:
                    t_norm = 2 * phase[i] - 1.0
                    bez = t_norm**3 + 3*(t_norm**2)*(1 - t_norm)
                    z_all[i] = 0.08 * (1.0 - bez)

            # 4.3) Build world‐frame foot positions for each step via cubic Bézier blend
            #       between p_foot_0 and p_foot_1
            # (a) Rotation mat at start and end in NumPy
            theta_0 = theta[0]; theta_f = theta[-1]
            c0, s0 = np.cos(theta_0), np.sin(theta_0)
            R0 = np.array([[ c0,  s0, 0],
                           [-s0,  c0, 0],
                           [  0,    0, 1]])
            c1_, s1_ = np.cos(theta_f), np.sin(theta_f)
            Rf = np.array([[ c1_,  s1_, 0],
                           [-s1_,  c1_, 0],
                           [   0,    0, 1]])
            p_com_0   = np.array([ x_t[0],    y_t[0],    0.0])
            p_com_end = np.array([ x_t[-1],   y_t[-1],   0.0])

            p_foot_0 = p_com_0 + (R0 @ default_foot_mat.T).T  # (4×3)
            p_foot_1 = p_com_end + (Rf @ default_foot_mat.T).T  # (4×3)

            # (b) For each i, foot_pos_world[i,:,:] = p_foot_0 + B[i]*(p_foot_1 - p_foot_0)
            #     where B[i] = phase[i]^3 + 3*phase[i]^2*(1-phase[i])
            B      = phase**3 + 3*(phase**2)*(1.0 - phase)     # shape (N,)
            foot_world_all = np.zeros((N, 4, 3))
            for i in range(N):
                foot_world_all[i,:,:] = p_foot_0 + B[i]*(p_foot_1 - p_foot_0)

            # 4.4) Now convert all these NumPy arrays into CasADi DM, ready for batch call
            # Inputs to the IK function should be shaped so that each column is one time step:
            #   • t_in:      (1 × N)
            #   • phase_in:  (1 × N)
            #   • z_in:      (1 × N)
            #   • foot_in:   (12 × N)  ← flatten 4×3 into 1×12, then stack N of them as columns
            #   • x_in:      (1 × N)
            #   • y_in:      (1 × N)
            #   • theta_in:  (1 × N)

            t_in     = ca.DM(ts).T                     # 1×N
            phase_in = ca.DM(phase).T                  # 1×N
            z_in     = ca.DM(z_all).T                  # 1×N
            # Flatten foot_world_all from (N,4,3) to (N,12) then transpose → (12 × N)
            # foot_flat   = foot_world_all.reshape((N, 12))  # shape (N, 12)
            # foot_in     = ca.DM(foot_flat).T               # 12×N
                        # Correct reshaping for CasADi .map(N) for input (4×3)
            foot_world_transposed = np.transpose(foot_world_all, (1, 2, 0))  # (4,3,N)
            foot_in = ca.DM(foot_world_transposed.reshape(4, 3*N))  # (4, 3×N) → correct!

            x_in        = ca.DM(x_t).T                      # 1×N
            y_in        = ca.DM(y_t).T                      # 1×N
            theta_in    = ca.DM(theta).T                    # 1×N

            # 4.5) Call the mapped IK function in parallel over all N steps
            ik_parallel = ik_func.map(N)  # apply single‐step IK to each of the N columns
            ts_DM, q_DM, foot_DM = ik_parallel(
                t_in, phase_in, z_in, foot_in, x_in, y_in, theta_in
            )
            # Convert outputs to NumPy
            ts_out     = np.array(ts_DM.full()).flatten()           # shape (N,)
            q_out      = np.array(q_DM.full()).T                    # shape (N, 12)
            foot_out   = np.array(foot_DM.full()).T.reshape((N, 4, 3))  # shape (N, 4, 3)

            # 4.6) Store into the large grids
            q_refs_all[ix, iy, iz, :, :]    = q_out        # (N,12)
            foot_refs_all[ix, iy, iz, :, :, :] = foot_out  # (N,4,3)

# -------------------------------
# 5) Apply 75% time‐shift and save everything (exactly as in generate_gait_libray)
# -------------------------------
ind_75 = int(N * 0.75)  # = 75

# (a) 75% shift for q_refs_all along the time axis (axis=3)
q_refs_rot = np.concatenate([
    q_refs_all[..., -ind_75:, :],     # last 25% (N−ind_75..N−1)
    q_refs_all[..., :ind_75, :]
], axis=3)  # shape: (vx_len, vy_len, wz_len, N, 12)

# (b) 75% shift for foot_refs_all along the time axis (axis=3)
foot_refs_rot = np.concatenate([
    foot_refs_all[..., -ind_75:, :, :],   # last 25%
    foot_refs_all[..., :ind_75, :, :],
], axis=3)  # shape: (vx_len, vy_len, wz_len, N, 4, 3)

# (c) Save numpy arrays in references/ folder
os.makedirs("references", exist_ok=True)
np.save("references/go2_reference_qs.npy", q_refs_rot)
np.save("references/go2_reference_foot_refs.npy", foot_refs_rot)

# -------------------------------
# 6) Reorder joints to “Isaac” convention and save
# -------------------------------
# Original MUJOCO joint ordering in generate_gait_libray:
joint_names_mujoco = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
]
# Isaac ordering desired (from the original code):
joint_names_isaac = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"
]
# Compute index mapping:
mujoco_to_isaac = [joint_names_mujoco.index(jn) for jn in joint_names_isaac]

# Apply that reorder to q_refs_rot: (vx_len, vy_len, wz_len, N, 12) → (vx_len, vy_len, wz_len, N, 12)
q_refs_np_isaac = q_refs_rot[..., mujoco_to_isaac]

# Save it
np.save("references/go2_reference_qs_isaac.npy", q_refs_np_isaac)

print("All references generated and saved in the `references/` folder.")
