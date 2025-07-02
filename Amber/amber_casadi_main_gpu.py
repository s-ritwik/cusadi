#!/usr/bin/env python3
import numpy as np
import pinocchio as pin
import casadi as ca
import time
import torch
from src import CusadiFunction   
reference_step = ca.Function.load("/home/s-ritwik/src/cusadi/reference_step.casadi")
print("CasADi version:", ca.__version__)

# Load the pre-generated Amber IK solver
reference_step = ca.Function.load("amber_reference_step.casadi")

# ----------------------------------------------------------------------
# 1) Robot & IK constants
# ----------------------------------------------------------------------
AMBER_URDF    = "Amber/amber_free.urdf"
FOOT_FRAMES   = ["left_toe", "right_toe"]
DAMPING_PIN   = 1e-4
IK_ITERS_PIN  = 20
IK_TOL_PIN    = 1e-6

# ----------------------------------------------------------------------
# 2) Build Pinocchio model for default foot‐poses & neutral q
# ----------------------------------------------------------------------
model_pin = pin.buildModelFromUrdf(AMBER_URDF)
data_pin  = model_pin.createData()

# Assume fixed‐base; neutral() yields exactly your 4 actuated joints
q_init = pin.neutral(model_pin)
# print(q_init)
q_init_dm = ca.DM(q_init.reshape(-1, 1))  # (4×1) for Amber

# Frame IDs for the two feet
frame_ids = [model_pin.getFrameId(name) for name in FOOT_FRAMES]

# Precompute nominal foot positions in world frame (NumPy)
pin.forwardKinematics(model_pin, data_pin, q_init)
pin.updateFramePlacements(model_pin, data_pin)
default_foot_positions = {
    fid: data_pin.oMf[fid].translation.copy() for fid in frame_ids
}

# ----------------------------------------------------------------------
# 3) Bézier swing helper (identical to Go2 version)
# ----------------------------------------------------------------------
def cubic_bezier_interpolation(z_start, z_end, t):
    t_clamped   = ca.fmax(0, ca.fmin(1, t))
    z_diff      = z_end - z_start
    bez         = t_clamped**3 + 3*(t_clamped**2)*(1 - t_clamped)
    return z_start + z_diff * bez

# ----------------------------------------------------------------------
# 4) Single‐gait generation for one v_x
# ----------------------------------------------------------------------
def generate_reference(
    v_x: float,
    foot_ids,
    default_foot_pos_dict,
    swing_height: float,
    T: float,
    N: int
):
    """
    Returns:
      ts_np      : (N,)      time samples
      q_ref_np   : (N×4)     IK joint‐refs
      foot_ref_np: (N×2×3)   foot positions in body‐frame
    """
    start = time.time()

    # 1) Pre-alloc CasADi DMs
    q_ref_cas         = ca.DM.zeros(N, 4)
    foot_ref_flat_cas = ca.DM.zeros(N, 2*3)

    # 2) Time vector & phase
    ts_np    = np.linspace(0, T, N)
    ts       = ca.DM(ts_np)
    phase_np = np.maximum(0.0, np.minimum(1.0, (ts_np/float(T) - 0.25)*2))
    phase    = ca.DM(phase_np)
    # print(phase)

    # 3) Swing height profile via Bézier
    mask    = phase_np <= 0.5
    z0      = cubic_bezier_interpolation(ca.DM(0),           ca.DM(swing_height), 2*phase)
    z1      = cubic_bezier_interpolation(ca.DM(swing_height), ca.DM(0),           2*phase - 1)
    z_array = ca.DM.zeros(N,1)
    for i in range(N):
        z_array[i] = ca.if_else(mask[i], z0[i], z1[i])
    # print(z_array)

    # 4) COM x‐trajectory
    x_array = v_x * ts  # straight‐line, no rotation
    # print(x_array)

    # 4.1) COM y‐trajectory (constant zero for planar robot)  ◀ ADDED
    y_array = ca.DM.zeros(N,1)                                               # ◀ ADDED

    # 5) Build default_foot_pos_mat (2×3) in DM
    default_foot_pos_mat = ca.DM.zeros(2,3)
    for idx, fid in enumerate(foot_ids):
        default_foot_pos_mat[idx,:] = ca.DM(default_foot_pos_dict[fid].reshape(3,))
    # print(default_foot_pos_mat)

    # Precompute COM‐start/end in world
    p_com_0   = ca.vertcat(x_array[0], y_array[0], ca.DM(0))                 # ◀ CHANGED
    p_com_end = ca.vertcat(x_array[-1], y_array[-1], ca.DM(0))               # ◀ CHANGED
    p_foot_0  = ca.repmat(p_com_0.T, 2, 1) + default_foot_pos_mat
    p_foot_1  = ca.repmat(p_com_end.T, 2, 1) + default_foot_pos_mat
    # print(p_foot_1)

    # # 6) Pre-build world‐foot stack for IK                         ◀ CHANGED
    foot_w_stack = ca.DM.zeros(6, N)                                         # ◀ ADDED
    for i in range(N):
        ph_i         = phase[i]
        foot_world_i = cubic_bezier_interpolation(p_foot_0, p_foot_1, ph_i)  # (2×3)
        foot_w_stack[:, i] = ca.reshape(foot_world_i, 6, 1)                  # ◀ ADDED
    # 7) ---- GPU‐batch IK for all N samples at once ----

    torch.cuda.synchronize()
    t0 = time.time()                         # optional timing
    q_prev = q_init_dm  # ◀ ADDED
    BATCH_SIZE = N
    # 7.1) Convert CasADi DMs to NumPy
    phase_np = phase.toarray().squeeze()           # (N,)
    x_np     = x_array.toarray().squeeze()         # (N,)
    y_np     = y_array.toarray().squeeze()         # (N,)
    z_np     = z_array.toarray().squeeze()         # (N,)
    foot_np  = foot_w_stack.toarray().T.copy()     # (N,6)

    # 7.2) Stack initial q into shape (N,4)
    q_init_np = np.tile(q_init.reshape(1,4), (N,1))  # (N,4)

    # 7.3) Move everything to CUDA tensors
    phase_t   = torch.from_numpy(phase_np).to('cuda', torch.double).unsqueeze(1)  # (N,1)
    x_t       = torch.from_numpy(x_np    ).to('cuda', torch.double).unsqueeze(1)
    y_t       = torch.from_numpy(y_np    ).to('cuda', torch.double).unsqueeze(1)
    z_t       = torch.from_numpy(z_np    ).to('cuda', torch.double).unsqueeze(1)
    foot_t    = torch.from_numpy(foot_np ).to('cuda', torch.double)              # (N,6)
    q_init_t  = torch.from_numpy(q_init_np).to('cuda', torch.double)             # (N,4)

    # 7.4) Wrap and launch the CasADi kernel on GPU
    fn = CusadiFunction(reference_step, N)
    fn.evaluate([phase_t, foot_t, x_t, y_t, z_t, q_init_t])

    # 7.5) Retrieve and reshape outputs
    q_batch     = fn.outputs_sparse[0]                           # (N,4)
    foot_batch  = fn.outputs_sparse[1]                           # (N,6)
    torch.cuda.synchronize()
    print("time:",(time.time()-t0)*1e3," ms")

    q_ref_np         = q_batch.cpu().numpy()                     # (N,4)
    foot_flat_np     = foot_batch.cpu().numpy()                  # (N,6)
    foot_ref_np      = foot_flat_np.reshape(N, 2, 3)            # (N,2,3)
    return ts_np, q_ref_np, foot_ref_np

# ----------------------------------------------------------------------
# 5) Generate over a grid of v_x only
# ----------------------------------------------------------------------
def generate_gait_library(
    v_xs_np: np.ndarray,
    swing_height: float = 0.1,
    T: float           = 0.4,
    N: int             = 100
):
    """
    Loops over v_xs_np → calls generate_reference → stacks & time‐shifts.
    Saves:
      references/amber_vxs.npy
      references/amber_reference_ts.npy
      references/amber_reference_qs.npy
      references/amber_reference_foot_refs.npy
    """
    vx_len = v_xs_np.size

    q_refs_list    = [None]*vx_len
    foot_refs_list = [None]*vx_len
    ts_out         = None

    for ix, vx in enumerate(v_xs_np):
        print(f"Generating Amber refs for v_x = {vx:.2f} m/s")
        ts_out, qref_np, foot_ref_np = generate_reference(
            vx,
            frame_ids,
            default_foot_positions,
            swing_height,
            T,
            N
        )
        q_refs_list[ix]    = qref_np.copy()
        foot_refs_list[ix] = foot_ref_np.copy()
    # print(q_refs_list[0])

    # Stack into full arrays
    q_refs_np    = np.stack(q_refs_list,    axis=0)  # (vx_len, N, 4)
    foot_refs_np = np.stack(foot_refs_list, axis=0)  # (vx_len, N, 2, 3)

    # 75% time‐shift
    ind_75       = int(N*0.75)
    q_refs_rot   = np.concatenate([q_refs_np[:, -ind_75:, :],
                                   q_refs_np[:, :ind_75, :]], axis=1)
    foot_refs_rot= np.concatenate([foot_refs_np[:, -ind_75:, :, :],
                                   foot_refs_np[:, :ind_75, :, :]], axis=1)

    # Make sure output folder exists
    import os
    os.makedirs("Amber/references", exist_ok=True)

    # Save arrays
    np.save("Amber/references/amber_vxs.npy",            v_xs_np)
    np.save("Amber/references/amber_reference_ts.npy",   ts_out)
    np.save("Amber/references/amber_reference_qs.npy",   q_refs_rot)
    np.save("Amber/references/amber_reference_foot_refs.npy", foot_refs_rot)

    print("→ All Amber references generated and saved.")

    return ts_out, q_refs_rot, foot_refs_rot

# ----------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a grid of forward speeds
    torch.cuda.synchronize()
    tstart=time.time()
    v_xs = np.linspace(-0.5,  0.5, 100)
    ts, q_refs, foot_refs = generate_gait_library(v_xs)
    torch.cuda.synchronize()
    print(f"GPU total time: {(time.time()-tstart)*1e3:.2f} ms")

