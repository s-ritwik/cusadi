import os
import numpy as np
import torch
import casadi as ca
from src import CusadiFunction  # Assuming this is available
import time
import pinocchio as pin

# Constants - must match original
GO2_URDF = "Go2_pinocchio/go2_original.urdf"
FOOT_FRAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
CASADI_FN_PATH = "casadi_functions/go2_one_step_ref.casadi"

# Load CasADi function
if not os.path.exists(CASADI_FN_PATH):
    raise FileNotFoundError(f"CasADi function not found at {CASADI_FN_PATH}")
one_step_ref = ca.Function.load(CASADI_FN_PATH)

# Build model
model_pin = pin.buildModelFromUrdf(GO2_URDF)
data_pin = model_pin.createData()
frame_ids = [model_pin.getFrameId(name) for name in FOOT_FRAMES]

# Neutral configuration
q_init = pin.neutral(model_pin)
q_init[:] = np.deg2rad([
    -5.7, 45.8, -86.0,
    +5.7, 45.8, -86.0,
    -5.7, 57.3, -86.0,
    +5.7, 57.3, -86.0
])

def generate_reference_parallel(v_x, v_y, w_z, default_foot_pos_dict, swing_height=0.08, T=0.4, N=100):
    """Generate reference trajectory matching original logic"""
    # 1. Create time vector
    ts = np.linspace(0, T, N)
    
    # 2. Compute heading and position (matches original)
    theta = w_z * ts
    if abs(w_z) < 0.01:
        x_t = v_x * ts
        y_t = v_y * ts
    else:
        x_t = (v_x * np.sin(w_z * ts) + v_y * (np.cos(w_z * ts) - 1)) / w_z
        y_t = (v_x * (1 - np.cos(w_z * ts)) + v_y * np.sin(w_z * ts)) / w_z

    # 3. Build default foot positions matrix
    default_foot_pos_mat = np.zeros((4, 3))
    for idx, fid in enumerate(frame_ids):
        default_foot_pos_mat[idx] = default_foot_pos_dict[fid]

    # 4. Rotation helper (matches original)
    def rot_z(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, s, 0],
            [-s, c, 0],
            [0, 0, 1]
        ])
    
    # 5. Compute R_start, R_end
    R_start = rot_z(theta[0])
    R_end = rot_z(theta[-1])
    
    # 6. Compute p_com_0 and p_com_end
    p_com_0 = np.array([x_t[0], y_t[0], 0])
    p_com_end = np.array([x_t[-1], y_t[-1], 0])
    
    # 7. Compute world foot positions at t=0 and t=T
    p_foot_0 = (R_start @ default_foot_pos_mat.T).T + p_com_0
    p_foot_1 = (R_end @ default_foot_pos_mat.T).T + p_com_end

    # 8. Prepare batch inputs
    phases = np.clip((ts / T - 0.25) * 2, 0, 1)
    p_foot_0_batch = np.tile(p_foot_0.flatten(), (N, 1))
    p_foot_1_batch = np.tile(p_foot_1.flatten(), (N, 1))
    x_i_batch = x_t.reshape(-1, 1)
    y_i_batch = y_t.reshape(-1, 1)
    theta_i_batch = theta.reshape(-1, 1)
    swing_batch = np.full((N, 1), swing_height)

    # 9. Convert to GPU tensors
    inputs = [
        torch.tensor(phases, device='cuda', dtype=torch.double),
        torch.tensor(p_foot_0_batch, device='cuda', dtype=torch.double),
        torch.tensor(p_foot_1_batch, device='cuda', dtype=torch.double),
        torch.tensor(x_i_batch, device='cuda', dtype=torch.double),
        torch.tensor(y_i_batch, device='cuda', dtype=torch.double),
        torch.tensor(theta_i_batch, device='cuda', dtype=torch.double),
        torch.tensor(swing_batch, device='cuda', dtype=torch.double)
    ]

    # 10. Execute in parallel
    batch_fn = CusadiFunction(one_step_ref, N)
    batch_fn.evaluate(inputs)
    
    # 11. Get results
    q_ref = batch_fn.outputs_sparse[0].cpu().numpy()
    foot_ref = batch_fn.outputs_sparse[1].cpu().numpy().reshape(N, 4, 3)
    
    return ts, q_ref, foot_ref

def generate_gait_library(v_xs, v_ys, w_zs, swing_height=0.08, T=0.4, N=100):
    """Generate full gait library matching original"""
    # Precompute default foot positions
    pin.forwardKinematics(model_pin, data_pin, q_init)
    pin.updateFramePlacements(model_pin, data_pin)
    default_foot_positions = {}
    for fid in frame_ids:
        default_foot_positions[fid] = data_pin.oMf[fid].translation.copy()
    
    # Initialize arrays
    vx_len, vy_len, wz_len = len(v_xs), len(v_ys), len(w_zs)
    all_ts = np.zeros((vx_len, vy_len, wz_len, N))
    all_q = np.zeros((vx_len, vy_len, wz_len, N, 12))
    all_foot = np.zeros((vx_len, vy_len, wz_len, N, 4, 3))
    
    # Generate references
    for i, vx in enumerate(v_xs):
        for j, vy in enumerate(v_ys):
            for k, wz in enumerate(w_zs):
                print(f"Generating for (vx={vx:.2f}, vy={vy:.2f}, wz={wz:.2f})")
                ts, q_ref, foot_ref = generate_reference_parallel(
                    vx, vy, wz, default_foot_positions, swing_height, T, N
                )
                all_ts[i, j, k] = ts
                all_q[i, j, k] = q_ref
                all_foot[i, j, k] = foot_ref
    
    # Time shift (75%) - matches original
    ind_75 = int(N * 0.75)
    q_rot = np.concatenate([all_q[..., -ind_75:, :], all_q[..., :ind_75, :]], axis=3)
    foot_rot = np.concatenate([all_foot[..., -ind_75:, :, :], all_foot[..., :ind_75, :, :]], axis=3)
    
    # Save results - matches original
    os.makedirs("references", exist_ok=True)
    np.save("references/go2_vxs.npy", v_xs)
    np.save("references/go2_vys.npy", v_ys)
    np.save("references/go2_wzs.npy", w_zs)
    np.save("references/go2_reference_ts.npy", all_ts)
    np.save("references/go2_reference_qs.npy", q_rot)
    np.save("references/go2_reference_foot_refs.npy", foot_rot)
    
    # Joint reordering for IsaacGym
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
    mujoco_to_isaac = [joint_names_mujoco.index(jn) for jn in joint_names_isaac]
    q_isaac = q_rot[..., mujoco_to_isaac]
    np.save("references/go2_reference_qs_isaac.npy", q_isaac)
    
    return all_ts, q_rot, foot_rot

if __name__ == "__main__":
    # Velocity grids - matches original
    v_xs = np.linspace(-1.0, 1.5, 11)
    v_ys = np.linspace(-0.75, 0.75, 7)
    w_zs = np.linspace(-0.5, 0.5, 5)
    
    # Generate gait library
    ts, q_refs, foot_refs = generate_gait_library(v_xs, v_ys, w_zs)
    print("Gait library generated and saved")