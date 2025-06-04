# go2_reference_cusadi.py

import os
import numpy as np
import pinocchio as pin
import torch
from casadi import *
from src import *  # from the cusadi repo

# ----------------------------------------------------------------------
# 1. Load the “single‐step” CasADi function and wrap in CusADi (once)
# ----------------------------------------------------------------------
N_GLOBAL = 100
# Make sure this path points to where generate_step.casadi is saved
fn_casadi_step = casadi.Function.load(
    os.path.join(CUSADI_FUNCTION_DIR, "generate_step.casadi")
)
fn_cusadi_step = CusadiFunction(fn_casadi_step, N_GLOBAL)

# We'll create a CusADiFunction dynamically once per batch size.
# But for convenience, define a helper to get a new CusADiFunction:

# ----------------------------------------------------------------------
# 2. Numeric Pinocchio setup (to compute default_foot_pos in world frame)
# ----------------------------------------------------------------------

GO2_URDF = "Go2_pinocchio/go2_original.urdf"
FOOT_FRAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

# Build numeric model once to extract default foot positions at neutral pose
model_pin  = pin.buildModelFromUrdf(GO2_URDF)
data_pin   = model_pin.createData()

# “Neutral” standing posture for Go2 (same as original Python)
q_init_numpy = np.deg2rad(np.array([
    -5.7,  45.8, -86.0,
    +5.7,  45.8, -86.0,
    -5.7,  57.3, -86.0,
    +5.7,  57.3, -86.0
]))  # shape (12,)

# Compute default foot positions in world frame at neutral pose
pin.forwardKinematics(model_pin, data_pin, q_init_numpy)
pin.updateFramePlacements(model_pin, data_pin)

frame_ids = [model_pin.getFrameId(name) for name in FOOT_FRAMES]
default_foot_pos_numpy = np.vstack([
    data_pin.oMf[fid].translation.copy() for fid in frame_ids
])  # shape (4,3)

# ----------------------------------------------------------------------
# 3. Function: gen_single_trajectory_cusadi
# ----------------------------------------------------------------------

def gen_single_trajectory_cusadi(
    v_x: float,
    v_y: float,
    w_z: float,
    swing_h: float,
    T: float,
    N: int
):
    """
    Compute one (ts, q_ref, foot_ref) trajectory of length N for given
    (v_x, v_y, w_z), using the GPU-parallel 'generate_step.casadi' function.

    Returns:
        ts      : (N,)       numpy array of time stamps
        q_ref   : (N, 12)    numpy array of joint angles (each row is 12)
        foot_ref: (N, 4, 3)  numpy array of foot positions in body frame
    """

    # 3.1. Build ts and p_com = [(x_t, y_t, theta_t)] in NumPy
    ts = np.linspace(0.0, T, N)        # shape (N,)
    theta_t = w_z * ts                 # shape (N,)

    # x(t), y(t) piecewise
    if abs(w_z) < 0.01:
        x_t = v_x * ts                 # shape (N,)
        y_t = v_y * ts
    else:
        x_t = (v_x * np.sin(w_z * ts) + v_y * (np.cos(w_z * ts) - 1.0)) / w_z
        y_t = (v_x * (1.0 - np.cos(w_z * ts)) + v_y * np.sin(w_z * ts)) / w_z

    # 3.2. Compute p_foot0 (4×3) at t=0
    theta0 = theta_t[0]
    c0     = np.cos(theta0)
    s0     = np.sin(theta0)
    R0     = np.array([[ c0, -s0, 0 ],
                       [ s0,  c0, 0 ],
                       [  0,   0, 1 ]])

    p_foot0 = np.zeros((4, 3))
    for k in range(4):
        p_world_k   = R0.dot(default_foot_pos_numpy[k])      # (3,)
        p_foot0[k]  = np.hstack(( x_t[0], y_t[0], 0.0 )) + p_world_k

    # 3.3. Compute p_foot1 (4×3) at t = T (index N-1)
    theta_end = theta_t[-1]
    ce        = np.cos(theta_end)
    se        = np.sin(theta_end)
    Re        = np.array([[ ce, -se, 0 ],
                          [ se,  ce, 0 ],
                          [  0,   0, 1 ]])

    p_foot1 = np.zeros((4, 3))
    for k in range(4):
        p_world_k_e  = Re.dot(default_foot_pos_numpy[k])      # (3,)
        p_foot1[k]   = np.hstack(( x_t[-1], y_t[-1], 0.0 )) + p_world_k_e

    # 3.4. Build per-step inputs for "generate_step" in NumPy
    phase_t = np.clip((ts / T - 0.25) * 2.0, 0.0, 1.0)   # shape (N,)

    # 3.5. Convert all arrays to PyTorch tensors on GPU
    device = "cuda"

    # Scalars per step: phase, x_i, y_i, theta_i  → shape (N,)
    phase_gpu  = torch.from_numpy(phase_t.astype(np.float64)).to(device)     # (N,)
    x_gpu      = torch.from_numpy(x_t.astype(np.float64)).to(device)         # (N,)
    y_gpu      = torch.from_numpy(y_t.astype(np.float64)).to(device)         # (N,)
    theta_gpu  = torch.from_numpy(theta_t.astype(np.float64)).to(device)     # (N,)

    # Constant (4×3) p_foot0, p_foot1 → broadcast to shape (N,4,3)
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

    # swing_h (scalar) → broadcast to (N,)
    swing_h_gpu = swing_h * torch.ones((N,), device=device, dtype=torch.double)

    # q_init (12×1) neutral posture → broadcast to shape (N,12)
    # q_init_gpu = (
    #     torch.from_numpy(q_init_numpy.astype(np.float64))
    #     .reshape(1, 12)
    #     .repeat(N, 1)
    #     .to(device)
    # )
    one_col = torch.from_numpy(q_init_numpy.astype(np.float64)).view(12, 1)  # (12,1)
    q_init_gpu = one_col.unsqueeze(0).repeat(N, 1, 1).to(device)  # shape (N, 12, 1)    
    # 3.6. Create or retrieve CusADiFunction for batch size N
    # fn_cusadi_step = make_cusadi_step_fn(N)

    # 3.7. Evaluate in parallel on GPU
    # Inputs order must match the signature of generate_step:
    #   [phase, x_i, y_i, theta, p_foot0, p_foot1, swing_h, q_init]
    inputs = [
        phase_gpu,   # (N,)
        x_gpu,       # (N,)
        y_gpu,       # (N,)
        theta_gpu,   # (N,)
        p0_tensor,   # (N,4,3)
        p1_tensor,   # (N,4,3)
        swing_h_gpu, # (N,)
        q_init_gpu   # (N,12)
    ]
    assert N == N_GLOBAL, "CusadiFunction batch size mismatch!"
    fn_cusadi_step.evaluate(inputs)

    # Retrieve GPU outputs:
    q_out_gpu     = fn_cusadi_step.outputs_sparse[0]  # (N,12)
    foot_flat_gpu = fn_cusadi_step.outputs_sparse[1]  # (N,12)

    # Move back to CPU NumPy
    q_ref_np       = q_out_gpu.cpu().numpy()            # (N,12)
    foot_flat_np   = foot_flat_gpu.cpu().numpy()        # (N,12)

    # Reshape foot_flat_np → (N,4,3)
    foot_ref_np    = foot_flat_np.reshape((N, 4, 3))

    return ts, q_ref_np, foot_ref_np


# ----------------------------------------------------------------------
# 4. Function: generate_gait_library_cusadi
# ----------------------------------------------------------------------

def generate_gait_library_cusadi(
    v_xs: np.ndarray,
    v_ys: np.ndarray,
    w_zs: np.ndarray,
    swing_h: float = 0.08,
    T: float = 0.4,
    N: int = 100
):
    """
    Loop over all combinations of v_xs, v_ys, w_zs, compute (ts, q_ref, foot_ref)
    for each triple using gen_single_trajectory_cusadi, and assemble into
    arrays just like the original generate_gait_libray.

    Returns:
      ts           : (N,) numpy array of time stamps
      q_refs       : (len(v_xs), len(v_ys), len(w_zs), N, 12)
      foot_refs    : (len(v_xs), len(v_ys), len(w_zs), N, 4, 3)
    """

    nx = v_xs.size
    ny = v_ys.size
    nz = w_zs.size

    # Preallocate arrays
    q_refs    = np.zeros((nx, ny, nz, N, 12), dtype=np.float64)
    foot_refs = np.zeros((nx, ny, nz, N, 4, 3), dtype=np.float64)
    ts_full   = None

    for ix, v_x in enumerate(v_xs):
        for iy, v_y in enumerate(v_ys):
            for iz, w_z in enumerate(w_zs):
                print(f"Generating reference for (v_x,v_y,w_z)=({v_x:.2f}, {v_y:.2f}, {w_z:.2f})")
                ts, q_ref_np, foot_ref_np = gen_single_trajectory_cusadi(
                    v_x, v_y, w_z,
                    swing_h,
                    T,
                    N
                )
                # ts will be the same for all combos; store once
                if ts_full is None:
                    ts_full = ts

                q_refs[ix, iy, iz, :, :]    = q_ref_np      # shape (N,12)
                foot_refs[ix, iy, iz, :, :, :] = foot_ref_np  # shape (N,4,3)
    # Pick a foot and time index:
    foot_idx = 0
    t_idx    = 42

    # Compare foot_ref_np[t_idx, foot_idx, :]  vs FK(q_ref_np[t_idx, :])
    # using Pinocchio forwardKinematics + updateFramePlacements.

    pin.forwardKinematics(model_pin, data_pin, q_ref_np[t_idx, :])
    pin.updateFramePlacements(model_pin, data_pin)

    p_fk = data_pin.oMf[frame_ids[foot_idx]].translation
    p_target = foot_ref_np[t_idx, foot_idx, :]

    err = p_target - p_fk
    norm_err = np.linalg.norm(err)

    if norm_err >= 1e-6:
        print(f"[WARN] Foot {FOOT_FRAMES[foot_idx]} at t[{t_idx}] did not converge: ‖err‖={norm_err:.3e}")
    return ts_full, q_refs, foot_refs


# ----------------------------------------------------------------------
# 5. Main block (identical usage to original script)
# ----------------------------------------------------------------------

if __name__ == "__main__":

    # Example grids (same as original)
    v_xs = np.linspace(-1.0, 1.5, 11)
    v_ys = np.linspace(-0.75, 0.75, 7)
    w_zs = np.linspace(-0.5, 0.5, 5)

    swing_h = 0.08
    T       = 0.4
    N       = 100

    ts, q_refs, foot_refs = generate_gait_library_cusadi(
        v_xs, v_ys, w_zs, swing_h, T, N
    )

    # Post‐processing (same as original):
    # Reorder from mid‐stance→mid‐stance to stance→swing
    ind_75 = int(ts.size * 0.75)

    q_refs = np.concatenate(
        (q_refs[..., -ind_75:, :], q_refs[..., :ind_75, :]),
        axis=-2
    )  # shape becomes (nx, ny, nz, N, 12)

    foot_refs = np.concatenate(
        (foot_refs[..., -ind_75:, :, :], foot_refs[..., :ind_75, :, :]),
        axis=-3
    )  # shape (nx, ny, nz, N, 4, 3)

    # Save to disk exactly as original
    np.save("references/go2_vxs.npy", v_xs)
    np.save("references/go2_vys.npy", v_ys)
    np.save("references/go2_wzs.npy", w_zs)
    np.save("references/go2_reference_ts.npy", ts)
    np.save("references/go2_reference_qs.npy", q_refs)
    np.save("references/go2_reference_foot_refs.npy", foot_refs)

    # Optionally, compute Isaac ordering just as original script did
    joint_names_isaac = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"
    ]
    joint_names_mujoco = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
    ]

    mujoco_to_isaac = [joint_names_mujoco.index(j) for j in joint_names_isaac]
    isaac_to_mujoco = [joint_names_isaac.index(j) for j in joint_names_mujoco]
    np.save("references/go2_reference_qs_isaac.npy",
            q_refs[..., mujoco_to_isaac])

    print("✅ All references generated and saved to 'references/' directory.")
