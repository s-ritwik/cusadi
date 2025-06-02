import mujoco
import numpy as np
import pinocchio as pin
from scipy.linalg import pinv

# ----------------------------------------------------------------------
# Pinocchio IK Model & constants (added for Pinocchio-based IK)
# ----------------------------------------------------------------------
GO2_URDF = "Go2_pinocchio/go2_corrected_fixed3.urdf"  # Path to the converted URDF for Go2
FOOT_FRAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
DAMPING_PIN = 1e-4          # Damped least-squares regularizer for Pinocchio IK
IK_ITERS_PIN = 100         # Max iterations per foot for Pinocchio IK
IK_TOL_PIN = 1e-6          # Position tolerance for Pinocchio IK

# Build the Pinocchio model and data once, to be reused in IK calls
model_pin = pin.buildModelFromUrdf(GO2_URDF)
data_pin = model_pin.createData()

# “Neutral” standing posture as a reasonable initial guess
q_init = pin.neutral(model_pin)
q_init[:] = np.deg2rad([
    -5.7, 45.8, -86.0,
    +5.7, 45.8, -86.0,
    -5.7, 57.3, -86.0,
    +5.7, 57.3, -86.0
])

# Precompute Pinocchio frame IDs for each foot
frame_ids = [model_pin.getFrameId(name) for name in FOOT_FRAMES]

# ----------------------------------------------------------------------
# Pinocchio-based IK solver (damped least squares, foot-by-foot)
# ----------------------------------------------------------------------
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
    """Multi-foot IK using Pinocchio; solves each foot in sequence."""
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
            if np.linalg.norm(err) < tol:
                break

            J6 = pin.computeFrameJacobian(
                model, data, q, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            J_pos = J6[:3, :]                             # take position rows
            J_dls = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T + damping * np.eye(3))
            q += J_dls @ err

        if np.linalg.norm(err) >= tol:
            print(f"[WARN] Foot {FOOT_FRAMES[k]} did not converge (‖err‖ = {np.linalg.norm(err):.3e} m)")

    return q

# ----------------------------------------------------------------------
# Cubic Bézier interpolation for swing foot height
# ----------------------------------------------------------------------
def cubic_bezier_interpolation(z_start, z_end, t):
    t = np.clip(t, 0, 1)
    z_diff = z_end - z_start
    bezier = t ** 3 + 3 * (t ** 2 * (1 - t))
    return z_start + z_diff * bezier

# ----------------------------------------------------------------------
# Replacement inverse_kinematics: use Pinocchio instead of MuJoCo
# ----------------------------------------------------------------------
def inverse_kinematics(foot_pos, foot_ids, data, model, tol=1e-9, max_iter=1000):
    """
    Calls the Pinocchio-based IK solver using the target foot positions.
    Parameters `foot_ids`, `data`, `model` (MuJoCo) are unused here but kept
    to preserve the original function signature. We convert foot_pos directly
    into the Pinocchio solver inputs.
    """
    # foot_pos is an array of shape (4, 3) in the robot's body frame
    targets = foot_pos.copy()
    # Solve IK with Pinocchio, ignoring MuJoCo-specific inputs
    q_sol = inverse_kinematics_pinocchio(
        targets_xyz=targets,
        foot_frame_ids=frame_ids,
        model=model_pin,
        data=data_pin,
        q_guess=q_init,
        tol=tol,
        max_iter=max_iter,
        damping=DAMPING_PIN
    )
    return q_sol

# ----------------------------------------------------------------------
# Generate reference trajectory (unchanged except IK call)
# ----------------------------------------------------------------------
def generate_reference(v_x, v_y, w_z, foot_ids, default_foot_pos, data, model, swing_height, T, N):
    q_ref = np.zeros((N, 12))
    foot_ref = np.zeros((N, 4, 3))
    default_foot_pos = np.array([default_foot_pos[foot_id] for foot_id in foot_ids])

    # First, integrate trajectory of CoM from mid-stance to mid-stance
    ts = np.linspace(0, T, N)
    theta = w_z * ts  # Heading integrates simply
    if abs(w_z) < 0.01:
        x_t = v_x * ts
        y_t = v_y * ts
    else:
        x_t = (v_x * np.sin(w_z * ts) + v_y * (np.cos(w_z * ts) - 1)) / w_z
        y_t = (v_x * (1 - np.cos(w_z * ts)) + v_y * np.sin(w_z * ts)) / w_z

    p_com = np.stack((x_t, y_t, theta), axis=1)

    p_foot_0 = np.hstack((p_com[0, :2], np.array(0))) + (
        np.array([
            [np.cos(p_com[0, 2]), -np.sin(p_com[0, 2]), 0],
            [np.sin(p_com[0, 2]), np.cos(p_com[0, 2]), 0],
            [0, 0, 1]
        ]) @ default_foot_pos.T
    ).T

    p_foot_1 = np.hstack((p_com[-1, :2], np.array(0))) + (
        np.array([
            [np.cos(p_com[-1, 2]), -np.sin(p_com[-1, 2]), 0],
            [np.sin(p_com[-1, 2]), np.cos(p_com[-1, 2]), 0],
            [0, 0, 1]
        ]) @ default_foot_pos.T
    ).T

    # Next, design trajectory for each foot
    for i in range(N):
        t = ts[i]
        phase = np.clip((t / T - 0.25) * 2, 0, 1)
        # First 25% of trajectory is last 50% of stance.
        # Next 50% is swing. Final 25% is stance.
        z = np.where(
            phase <= 0.5,
            cubic_bezier_interpolation(0, swing_height, 2 * phase),
            cubic_bezier_interpolation(swing_height, 0, 2 * phase - 1)
        )
        # Spatial positions in global frame
        foot_pos = cubic_bezier_interpolation(p_foot_0, p_foot_1, phase)
        # Convert to body frame
        foot_pos = (
            np.array([
                [np.cos(p_com[i, 2]), np.sin(p_com[i, 2]), 0],
                [-np.sin(p_com[i, 2]), np.cos(p_com[i, 2]), 0],
                [0, 0, 1]
            ]) @ (foot_pos - np.hstack((p_com[i, :2], np.array(0)))).T
        ).T
        foot_pos[:, -1] += z

        foot_ref[i, :, :] = foot_pos
        q_ref[i, :] = inverse_kinematics(foot_pos, foot_ids, data, model)
    return ts, q_ref, foot_ref

# ----------------------------------------------------------------------
# Updated generate_gait_libray() and main block to use Pinocchio instead of MuJoCo
# ----------------------------------------------------------------------

def generate_gait_libray(v_xs, v_ys, w_zs, swing_height=0.08, T=0.4, N=100):
    # Use Pinocchio for kinematics
    model = model_pin
    data = data_pin

    # Set robot to neutral pose and compute default foot positions in world frame
    q = q_init.copy()
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    default_foot_positions = {}
    for i, fid in enumerate(frame_ids):
        default_foot_positions[fid] = data.oMf[fid].translation.copy()

    # foot_ids now correspond to frame_ids
    foot_ids = frame_ids.copy()

    q_refs = np.zeros((v_xs.size, v_ys.size, w_zs.size, N, 12))
    foot_refs = np.zeros((v_xs.size, v_ys.size, w_zs.size, N, 4, 3))
    ts = None

    for ix, v_x in enumerate(v_xs):
        for iy, v_y in enumerate(v_ys):
            for iz, w_z in enumerate(w_zs):
                print(f"Generating reference {v_x}, {v_y}, {w_z}")
                ts, qref, foot_ref = generate_reference(
                    v_x, v_y, w_z,
                    foot_ids,
                    default_foot_positions,
                    None,    # MuJoCo data/model arguments unused in Pinocchio IK
                    None,
                    swing_height,
                    T,
                    N
                )
                q_refs[ix, iy, iz, :] = qref
                foot_refs[ix, iy, iz, :] = foot_ref

    return ts, q_refs, foot_refs


if __name__ == "__main__":

    v_xs = np.linspace(-1.0, 1.5, 11)
    v_ys = np.linspace(-0.75, 0.75, 7)
    w_zs = np.linspace(-0.5, 0.5, 5)

    ts, q_refs, foot_refs = generate_gait_libray(v_xs, v_ys, w_zs)

    # q_refs is currently mid-stance to mid-stance. We want stance -> swing
    ind_75 = int(ts.size * 0.75)
    q_refs = np.concatenate(
        (q_refs[..., -ind_75:, :], q_refs[..., :ind_75, :]),
        axis=-2
    )
    foot_refs = np.concatenate(
        (foot_refs[..., -ind_75:, :, :], foot_refs[..., :ind_75, :, :]),
        axis=-3
    )

    np.save("references/go2_vxs.npy", v_xs)
    np.save("references/go2_vys.npy", v_ys)
    np.save("references/go2_wzs.npy", w_zs)
    np.save("references/go2_reference_ts.npy", ts)
    np.save("references/go2_reference_qs.npy", q_refs)
    np.save("references/go2_reference_foot_refs.npy", foot_refs)

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

    mujoco_to_isaac = [joint_names_mujoco.index(joint_name) for joint_name in joint_names_isaac]
    isaac_to_mujoco = [joint_names_isaac.index(joint_name) for joint_name in joint_names_mujoco]
    np.save("references/go2_reference_qs_isaac.npy", q_refs[..., mujoco_to_isaac])
