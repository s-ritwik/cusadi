# import mujoco
import numpy as np
import pinocchio as pin 
import torch
import time
from scipy.linalg import pinv
import casadi as ca
print(ca.__version__)
reference_step = ca.Function.load("/home/s-ritwik/src/cusadi/reference_step.casadi")
from src import CusadiFunction   


N = 100  # or whatever you used in generate_reference
BATCH_SIZE = N

# ----------------------------------------------------------------------
# Pinocchio IK Model & constants (unchanged)
# ----------------------------------------------------------------------
GO2_URDF = "Go2_pinocchio/go2_original.urdf"
FOOT_FRAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
DAMPING_PIN = 1e-4
IK_ITERS_PIN = 100
IK_TOL_PIN = 1e-6

# Build Pinocchio model from URDF
model_pin = pin.buildModelFromUrdf(GO2_URDF)
data_pin = model_pin.createData()
# model_pin = pin.buildModelFromUrdf(GO2_URDF, pin.JointModelFreeFlyer(), ca.SX)
# data_pin = model_pin.createData()
# A reasonable “neutral” pose for the Go2 quadruped (in radians)
q_init = pin.neutral(model_pin)
q_init[:] = np.deg2rad([
    -5.7, 45.8, -86.0,
    +5.7, 45.8, -86.0,
    -5.7, 57.3, -86.0,
    +5.7, 57.3, -86.0
])
frame_ids = [model_pin.getFrameId(name) for name in FOOT_FRAMES]

q_init_dm      = ca.DM(q_init.reshape(12, 1))     # neutral pose passed every call

# ----------------------------------------------------------------------
def cubic_bezier_interpolation(z_start, z_end, t):
    """
    Input: z_start, z_end are CasADi DM scalars; t is a CasADi DM scalar (or vector).
    Output: CasADi DM of same shape as t. Basis: B(t) = t^3 + 3 t^2 (1−t).
    """
    t_clamped = ca.fmax(0, ca.fmin(1, t))
    z_diff = z_end - z_start
    bezier_basis = t_clamped**3 + 3*(t_clamped**2)*(1 - t_clamped)
    return z_start + z_diff * bezier_basis


# ----------------------------------------------------------------------
# “→ replaced with CasADi” Generate reference trajectory
# ----------------------------------------------------------------------
def generate_reference(
    v_x, v_y, w_z,
    foot_ids,                  # list of 4 frame_ids (Pinocchio)
    default_foot_pos_dict,     # dict: frame_id → NumPy(3,)
    swing_height,
    T,
    N
):
    # ) Rotation helper (CasADi DM → 3×3)
    def rot_z_dm(angle_dm):
        c = ca.cos(angle_dm)
        s = ca.sin(angle_dm)
        return ca.horzcat(
            ca.vertcat(c,  s,  ca.DM(0)),
            ca.vertcat(-s, c,  ca.DM(0)),
            ca.vertcat(ca.DM(0), ca.DM(0), ca.DM(1))
        )
    """
    Now: everything in this function → CasADi DM (for positions, interpolation, etc.).
    Whenever we need to call Pinocchio IK, we convert the relevant DM back to NumPy.
    Returns:
      ts_np      : NumPy array of shape (N,)
      q_ref_np   : NumPy array of shape (N × 12)
      foot_ref_np: NumPy array of shape (N × 4 × 3)
    """
    # 1) Pre‐allocate a CasADi DM for q‐refs: shape (N × 12)
    q_ref_cas = ca.DM.zeros(N, 12)

    # 2) Pre‐allocate a CasADi DM for “flattened” foot positions: (N × (4×3))
    foot_ref_flat_cas = ca.DM.zeros(N, 4*3)

    # 3) Build CasADi time vector ts (N×1)
    #  Build a Python list (or NumPy) of scalar times [0, T] with N points:
    ts_np = np.linspace(0, T, N)                 # NumPy array (N,)
    ts    = ca.DM(ts_np)    
    # 2) Compute the “phase” for each t_i (NumPy→DM):
    phase_np = np.maximum(0.0, np.minimum(1.0, (ts_np/float(T) - 0.25)*2))  # (N,)
    phase    = ca.DM(phase_np)                    # DM (N,)

    # 3) Compute “z_i” for each phase_i via Bézier.  We can do this in DM‐vector form:

    #    First split into two halves:
    mask = phase_np <= 0.5                         # boolean mask (N,)
    # Build DM versions of z when phase≤0.5 and when phase>0.5:
    z0 = cubic_bezier_interpolation(ca.DM(0), ca.DM(swing_height), 2*phase)     # DM (N,)
    z1 = cubic_bezier_interpolation(ca.DM(swing_height), ca.DM(0), 2*phase - 1)  # DM (N,)

    # 4) Heading θ(t) = w_z * t  (DM)
    theta_array = w_z * ts             # DM (N,)
    z_array = ca.DM.zeros(N, 1)
    for i in range(N):
        z_array[i] = ca.if_else(mask[i], z0[i], z1[i])   # DM scalar each

    if abs(w_z) < 0.01:
        x_array = v_x * ts    # DM (N,)
        y_array = v_y * ts    # DM (N,)
    else:
        x_array = (v_x * ca.sin(w_z*ts) + v_y*(ca.cos(w_z*ts) - 1)) / w_z
        y_array = (v_x*(1 - ca.cos(w_z*ts)) + v_y * ca.sin(w_z*ts)) / w_z

    # 6) Build default foot positions (4×3) as a single CasADi DM
    default_foot_pos_mat = ca.DM.zeros(4, 3)
    for idx, fid in enumerate(foot_ids):
        default_foot_pos_mat[idx, :] = ca.DM(default_foot_pos_dict[fid].reshape((3,)))


    # Pre‐compute R_start, R_end in CasADi
    theta_start = theta_array[0]     # DM scalar
    theta_end   = theta_array[-1]    # DM scalar
    R_start = rot_z_dm(theta_start)  # DM (3×3)
    R_end   = rot_z_dm(theta_end)    # DM (3×3)


    # Build p_com_0 and p_com_end (3×1 each) as CasADi DM
    # p_com_0   = ca.vertcat(ts[0]*0 + 0 + x_t[0]*0 + y_t[0]*0,  # trick: just build (x(0),y(0),0)
    #                       ts[0]*0 + 0 + x_t[0] - x_t[0], 
    #                       ca.DM(0))  # actually easiest: p_com_0 = [x_t[0], y_t[0], 0]
    # But better to write explicitly:
    p_com_0   = ca.vertcat(x_array[0], y_array[0], ca.DM(0))   # DM (3×1)
    p_com_end = ca.vertcat(x_array[-1], y_array[-1], ca.DM(0))# DM (3×1)

    p_foot_0 = ca.repmat(p_com_0.T, 4, 1) + ((R_start @ default_foot_pos_mat.T).T)  # DM (4×3)
    p_foot_1 = ca.repmat(p_com_end.T, 4, 1) + ((R_end   @ default_foot_pos_mat.T).T)  # DM (4×3)

    # 8) Now precompute “foot_w_stack” as a DM of shape (12×N):

    #    For each i, foot_pos_world_i = Bézier(p_foot_0, p_foot_1, phase_i).
    #    We can do that in a loop or vectorized. Easiest: loop once:

    foot_w_stack = ca.DM.zeros(12, N)   # each column is foot_w for that i
    for i in range(N):
        # Extract the scalar phase_i:
        ph_i = phase[i]                             # DM scalar
        # 4×3 world positions at time i:
        foot_world_i = cubic_bezier_interpolation(p_foot_0, p_foot_1, ph_i)  # DM (4×3)
        # Flatten to 12×1 and store as column i:
        foot_w_stack[:, i] = ca.reshape(foot_world_i, 12, 1)

    


    # 9) Loop over each time index i to fill q_ref_cas and foot_ref_flat_cas
    torch.cuda.synchronize()
    t0 = time.time()                         # optional timing

    BATCH_SIZE = N
    fn_cusadi_ref = CusadiFunction(reference_step, BATCH_SIZE)         # wrap the .casadi kernel
    # --- DM  → NumPy -------------------------------------------------------------
    phase_np   = phase.toarray().squeeze()          # (N,)
    x_np       = x_array.toarray().squeeze()
    y_np       = y_array.toarray().squeeze()
    theta_np   = theta_array.toarray().squeeze()
    z_np       = z_array.toarray().squeeze()
    foot_np    = foot_w_stack.toarray().T.copy()           # (N,12)  ← transpose once here

    # --- NumPy → CUDA tensors ----------------------------------------------------
    ph_torch     = torch.from_numpy(phase_np ).to('cuda', torch.double).unsqueeze(1)   # (N,1)
    x_torch      = torch.from_numpy(x_np     ).to('cuda', torch.double).unsqueeze(1)
    y_torch      = torch.from_numpy(y_np     ).to('cuda', torch.double).unsqueeze(1)
    theta_torch  = torch.from_numpy(theta_np ).to('cuda', torch.double).unsqueeze(1)
    z_torch      = torch.from_numpy(z_np     ).to('cuda', torch.double).unsqueeze(1)
    foot_torch   = torch.from_numpy(foot_np  ).to('cuda', torch.double)     
    q_init_np    = np.tile(q_init.reshape(1,12), (N,1))                                                      # (N,12)
    q_init_torch = torch.from_numpy(q_init_np).to('cuda', torch.double)

    # --- run the kernel ---------------------------------------------------------
    fn_cusadi_ref.evaluate([
        ph_torch,           # (N,1)
        foot_torch,         # (N,12)
        x_torch,            # (N,1)
        y_torch,            # (N,1)
        theta_torch,        # (N,1)
        z_torch,            # (N,1)
        q_init_torch        # (N,12)
    ])

    q_batch_torch      = fn_cusadi_ref.outputs_sparse[0]        # (N,12)
    foot_batch_torch   = fn_cusadi_ref.outputs_sparse[1]        # (N,12)

    torch.cuda.synchronize()
    print(f"GPU batch time: {(time.time()-t0)*1e3:.2f} ms")

    # --- copy back to CPU NumPy --------------------------------------------------
    q_ref_np          = q_batch_torch.cpu().numpy()             # (N,12)
    foot_ref_np_flat  = foot_batch_torch.cpu().numpy()          # (N,12)
    foot_ref_np       = foot_ref_np_flat.reshape(N, 4, 3)       # (N,4,3)

    # 10) Convert CasADi → NumPy for returns:
    # ts_np = np.linspace(0, T, N)  # or ts.toarray().flatten()
    # q_ref_np = q_ref_cas.toarray()       # shape (N,12)
    # foot_ref_np = foot_ref_flat_cas.toarray().reshape((N,4,3))

    return ts_np, q_ref_np, foot_ref_np
# ----------------------------------------------------------------------
# “→ replaced with CasADi” generate_gait_libray (collect references, stack in NumPy)
# ----------------------------------------------------------------------
def generate_gait_libray(v_xs_np, v_ys_np, w_zs_np, swing_height=0.08, T=1, N=100):
    """
    - Builds default foot positions (NumPy) via Pinocchio.
    - For each (v_x, v_y, w_z), calls generate_reference (CasADi inside).
    - Collects each qref/foot_ref as NumPy arrays and stacks at the end.
    Returns:
       ts_out        : NumPy (N,)
       q_refs_np     : NumPy (|v_x| × |v_y| × |w_z| × N × 12)
       foot_refs_np  : NumPy (|v_x| × |v_y| × |w_z| × N × 4 × 3)
    """
    # 1) Build default foot positions in world frame (NumPy) using Pinocchio
    pin.forwardKinematics(model_pin, data_pin, q_init)
    pin.updateFramePlacements(model_pin, data_pin)
    default_foot_positions = {}
    for idx, fid in enumerate(frame_ids):
        default_foot_positions[fid] = data_pin.oMf[fid].translation.copy()

    foot_ids = frame_ids.copy()

    vx_len = v_xs_np.size
    vy_len = v_ys_np.size
    wz_len = w_zs_np.size

    # 2) Pre‐allocate Python lists to collect “one‐reference‐each”
    q_refs_list = [[[None for _ in range(wz_len)] for _ in range(vy_len)] for _ in range(vx_len)]
    foot_refs_list = [[[None for _ in range(wz_len)] for _ in range(vy_len)] for _ in range(vx_len)]
    ts_out = None

    # 3) Loop over all combinations:
    for ix, vx in enumerate(v_xs_np):
        for iy, vy in enumerate(v_ys_np):
            for iz, wz in enumerate(w_zs_np):
                print(f"Generating reference for (v_x, v_y, w_z)=({vx:.2f}, {vy:.2f}, {wz:.2f})")
                ts_out, qref_np, foot_ref_np = generate_reference(
                    vx, vy, wz,
                    foot_ids,
                    default_foot_positions,
                    swing_height,
                    T,
                    N
                )
                # Each qref_np is NumPy shape (N,12); foot_ref_np is (N,4,3)
                q_refs_list[ix][iy][iz] = qref_np.copy()
                foot_refs_list[ix][iy][iz] = foot_ref_np.copy()

    # 4) Now stack into big NumPy arrays:
    q_refs_np = np.zeros((vx_len, vy_len, wz_len, N, 12))
    foot_refs_np = np.zeros((vx_len, vy_len, wz_len, N, 4, 3))

    for ix in range(vx_len):
        for iy in range(vy_len):
            for iz in range(wz_len):
                q_refs_np[ix,iy,iz,:,:] = q_refs_list[ix][iy][iz]
                foot_refs_np[ix,iy,iz,:,:,:] = foot_refs_list[ix][iy][iz]

    # 5) Perform the 75% time‐shift (still NumPy)
    ind_75 = int(N * 0.75)
    # q‐refs: shape (..., N, 12)
    q_refs_rot = np.concatenate([
        q_refs_np[..., -ind_75:, :],
        q_refs_np[..., :ind_75, :]
    ], axis=3)  # shift along the “time” dimension

    foot_refs_rot = np.concatenate([
        foot_refs_np[..., -ind_75:, :, :],
        foot_refs_np[..., :ind_75, :, :, :]
    ], axis=3)  # shift along time axis

    # 6) Save all grids & references as before:
    np.save("references/go2_vxs.npy", v_xs_np)
    np.save("references/go2_vys.npy", v_ys_np)
    np.save("references/go2_wzs.npy", w_zs_np)
    np.save("references/go2_reference_ts.npy", ts_out)

    np.save("references/go2_reference_qs.npy", q_refs_rot)
    np.save("references/go2_reference_foot_refs.npy", foot_refs_rot)

    # Re‐order joints into “Isaac” format, same as original:
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

    # Re‐order q_refs_rot into Isaac joint ordering
    q_refs_np_isaac = q_refs_rot[..., mujoco_to_isaac]
    np.save("references/go2_reference_qs_isaac.npy", q_refs_np_isaac)

    return ts_out, q_refs_np, foot_refs_np

# ----------------------------------------------------------------------
# Main block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define velocity grids in NumPy
    v_xs = np.linspace(-1.0, 1.5, 11)
    v_ys = np.linspace(-0.75, 0.75, 7)
    w_zs = np.linspace(-0.5, 0.5, 5)

    ts, q_refs, foot_refs = generate_gait_libray(v_xs, v_ys, w_zs)
    print("All references generated and saved (using CasADi inside).")
