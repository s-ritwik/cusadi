# import mujoco
import numpy as np
import pinocchio as pin
from scipy.linalg import pinv
import casadi as ca

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



def inverse_kinematics_pinocchio(
    targets_xyz: np.ndarray,
    foot_frame_ids,
    model,
    data,
    q_guess=None,
    tol=IK_TOL_PIN,
    max_iter=IK_ITERS_PIN,
    damping=DAMPING_PIN,
):
    """
    Multi‐foot IK using Pinocchio, solving feet one‐by‐one in sequence.
    ‘targets_xyz’ has shape (4,3) as a NumPy array. Returns a NumPy vector of length 12.
    """
    if q_guess is None:
        q = q_init.copy()
    else:
        q = q_guess.copy()

    for k, fid in enumerate(foot_frame_ids):
        tgt = targets_xyz[k]
        for _ in range(max_iter):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)

            p_cur = data.oMf[fid].translation
            err = tgt - p_cur
            if ca.norm_2(err) < tol:
                break

            J6 = pin.computeFrameJacobian(
                model, data, q, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            # print("J6 computed")
            J_pos = J6[:3, :]
            # J_dls = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T + damping * np.eye(3))
            # J_dls = J_pos.T @ ca.inv(J_pos @ J_pos.T + damping * ca.MX.eye(3))
            J_dls = (ca.DM(J_pos).T @ ca.inv(ca.DM(J_pos) @ ca.DM(J_pos).T
                                 + damping * ca.DM.eye(3))).full()
            # print(type(q))
            q += J_dls @ err
            # print(type(q))

        if ca.norm_2(err) >= tol:
            print(f"[WARN] Foot {FOOT_FRAMES[k]} did not converge (‖err‖ = {ca.norm_2(err):.3e} m)")

    return q

# ----------------------------------------------------------------------
# “→ replaced with CasADi” Cubic Bézier interpolation for swing foot height
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
# Replacement inverse_kinematics: accepts CasADi DM but converts to NumPy
# ----------------------------------------------------------------------
def inverse_kinematics(foot_pos_casadi: ca.DM):
    """
    foot_pos_casadi: a CasADi DM of shape (4×3), encoding world‐frame foot positions per row.
    Convert to NumPy, call Pinocchio, return a length‐12 NumPy q.
    """
    foot_pos_np = foot_pos_casadi.toarray().reshape((4,3))
    q_sol = inverse_kinematics_pinocchio(
        targets_xyz=foot_pos_np,
        foot_frame_ids=frame_ids,
        model=model_pin,
        data=data_pin,
        q_guess=q_init,
        tol=IK_TOL_PIN,
        max_iter=IK_ITERS_PIN,
        damping=DAMPING_PIN
    )
    return q_sol  # NumPy array of length 12

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
    ts = ca.linspace(0, T, N)  # DM of size (N,)

    # 4) Heading θ(t) = w_z * t  (DM)
    theta = w_z * ts             # DM (N,)

    # 5) Compute x(t), y(t) in CasADi
    if abs(w_z) < 0.01:
        x_t = v_x * ts           # DM (N,)
        y_t = v_y * ts           # DM (N,)
    else:
        x_t = (v_x * ca.sin(w_z * ts) + v_y * (ca.cos(w_z * ts) - 1)) / w_z
        y_t = (v_x * (1 - ca.cos(w_z * ts)) + v_y * ca.sin(w_z * ts)) / w_z

    # 6) Build default foot positions (4×3) as a single CasADi DM
    default_foot_pos_mat = ca.DM.zeros(4, 3)
    for idx, fid in enumerate(foot_ids):
        default_foot_pos_mat[idx, :] = ca.DM(default_foot_pos_dict[fid].reshape((3,)))

    # 7) Rotation helper (CasADi DM → 3×3)
    def rot_z_dm(angle_dm):
        c = ca.cos(angle_dm)
        s = ca.sin(angle_dm)
        return ca.horzcat(
            ca.vertcat(c,  s,  ca.DM(0)),
            ca.vertcat(-s, c,  ca.DM(0)),
            ca.vertcat(ca.DM(0), ca.DM(0), ca.DM(1))
        )


    # Pre‐compute R_start, R_end in CasADi
    theta_start = theta[0]     # DM scalar
    theta_end   = theta[-1]    # DM scalar
    R_start = rot_z_dm(theta_start)  # DM (3×3)
    R_end   = rot_z_dm(theta_end)    # DM (3×3)

    # Build p_com_0 and p_com_end (3×1 each) as CasADi DM
    # p_com_0   = ca.vertcat(ts[0]*0 + 0 + x_t[0]*0 + y_t[0]*0,  # trick: just build (x(0),y(0),0)
    #                       ts[0]*0 + 0 + x_t[0] - x_t[0], 
    #                       ca.DM(0))  # actually easiest: p_com_0 = [x_t[0], y_t[0], 0]
    # But better to write explicitly:
    p_com_0 = ca.vertcat(x_t[0], y_t[0], ca.DM(0))
    p_com_end = ca.vertcat(x_t[-1], y_t[-1], ca.DM(0))

    # 8) Compute world‐frame foot positions at t=0 and t=T:
    #    p_foot_0 = p_com_0 + R_start @ default_foot_pos_mat^T  (then transpose)
    p_foot_0 = ca.repmat(p_com_0.T, 4, 1) + ( (R_start @ default_foot_pos_mat.T).T )
    p_foot_1 = ca.repmat(p_com_end.T, 4, 1) + ( (R_end   @ default_foot_pos_mat.T).T )

    # 9) Loop over each time index i to fill q_ref_cas and foot_ref_flat_cas
    for i in range(N):
        # 9.1) Phase logic (pure Python, since we need a float to do “phase”):
        t_i = float(ts[i])  # extract a Python float from DM
        phase = max(0.0, min(1.0, (t_i / float(T) - 0.25) * 2))

        if phase <= 0.5:
            z_i_dm = cubic_bezier_interpolation(ca.DM(0), ca.DM(swing_height), 2 * phase)
        else:
            z_i_dm = cubic_bezier_interpolation(ca.DM(swing_height), ca.DM(0), 2 * phase - 1)

        z_i = float(z_i_dm)  # convert DM scalar → float (if you're staying numeric)
                
        # 9.3) Interpolate foot position in world (CasADi DM):  
        #      foot_pos_world_cas = p_foot_0 + (p_foot_1 - p_foot_0)*phase
        foot_pos_world_cas =cubic_bezier_interpolation(p_foot_0, p_foot_1, phase)  # DM (4×3)

        # 9.4) Convert COM position at step i to floats for “body‐frame” rotation
        x_i = float(x_t[i])  
        y_i = float(y_t[i])
        θ_i = float(theta[i])

        # Build a CasADi 3×3 “world→body” rotation matrix at this instant:
        c_i = ca.cos(θ_i)
        s_i = ca.sin(θ_i)
        body_rot = ca.vertcat(
            ca.horzcat(c_i,  s_i,   0),
            ca.horzcat(-s_i, c_i,   0),
            ca.horzcat(  0,    0,   1)
        )

        # 9.5) For each foot, convert foot_pos_world_cas[f] → CasADi DM (3×1),
        #      subtract CoM([x_i, y_i, 0]), rotate by body_rot, then add swing z.
        foot_pos_body_cas = ca.DM.zeros(4, 3)   # 4×3 CasADi DM
        for f in range(4):
            # Extract the f-th row of foot_pos_world_cas as a 3×1 CasADi column
            pw = foot_pos_world_cas[f, :].T       # shape (3,1)

            # Compute relative vector in world frame (3×1)
            rel = pw - ca.vertcat(x_i, y_i, 0)

            # Rotate into body frame, then add vertical swing z_i
            pb = body_rot @ rel                   # shape (3,1)
            pb = pb + ca.vertcat(0, 0, z_i)

            # Store as a 1×3 row in foot_pos_body_cas
            foot_pos_body_cas[f, :] = pb.T        # pb.T is shape (1,3)

        # 9.6) Flatten the 4×3 CasADi DM into a 1×12 and store in foot_ref_flat_cas
        foot_ref_flat_cas[i, :] = ca.reshape(foot_pos_body_cas, 1, 12)

        # 9.7) Call IK using the CasADi DM directly
        #      (now ‘foot_pos_body_cas’ is already a (4×3) DM)
        q_cas = inverse_kinematics(foot_pos_body_cas)   # returns a CasADi DM of shape (12,)
        q_ref_cas[i, :] = ca.DM(q_cas.reshape((1,12)))  # store back as DM(1×12)

    # 10) Convert CasADi → NumPy for returns:
    ts_np = np.linspace(0, T, N)  # or ts.toarray().flatten()
    q_ref_np = q_ref_cas.toarray()       # shape (N,12)
    foot_ref_np = foot_ref_flat_cas.toarray().reshape((N,4,3))

    return ts_np, q_ref_np, foot_ref_np

# ----------------------------------------------------------------------
# “→ replaced with CasADi” generate_gait_libray (collect references, stack in NumPy)
# ----------------------------------------------------------------------
def generate_gait_libray(v_xs_np, v_ys_np, w_zs_np, swing_height=0.08, T=0.4, N=100):
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
