import numpy as np
import casadi as ca
import pinocchio as pin

# ———————— 1) Load IK model ————————
# Adjust the filename/path if needed
ik_fun = ca.Function.load("amber_reference_step.casadi")

# ———————— 2) Load FK model ————————
URDF = "Amber/amber_free.urdf"
model = pin.buildModelFromUrdf(URDF)
data  = model.createData()
# Frame names must match those in your URDF
foot_frames = ["left_toe", "right_toe"]
frame_ids   = [model.getFrameId(f) for f in foot_frames]

def ik_fk_demo(foot_x,       # np.ndarray(2,) = [xL, xR]
               z_swing=0.0,
               q_init=None,
               noise_std=1e-2):
    """
    Args:
        foot_x    : 2-element array of foot x-offsets (left, right)
        z_swing   : swing height offset
        q_init    : 4-element initial guess for IK (default zeros)
        noise_std : standard deviation of Gaussian noise on joints

    Returns:
        q_ref   : 4-element IK solution
        q_noisy : q_ref with added noise
        foot_fk : 6-element FK foot positions [x,y,z]_L, [x,y,z]_R
    """
    # --- prepare CasADi inputs ---
    phase_ca   = ca.DM(0)                             # single-step phase
    foot_x_ca  = ca.DM(foot_x.reshape(2, 1))          # 2×1
    zs_ca      = ca.DM(z_swing)                       # scalar
    if q_init is None:
        q_init = np.zeros(4)
    q_cur = ca.DM(q_init.reshape(4, 1))               # 4×1

    # --- run IK ---
    q_ref_ca, _ = ik_fun(phase_ca, foot_x_ca, zs_ca, q_cur)
    q_ref = np.array(q_ref_ca).flatten()              # (4,)

    # --- add noise ---
    q_noisy = q_ref + np.random.randn(4) * noise_std

    # --- run FK in Pinocchio ---
    pin.forwardKinematics(model, data, q_noisy)
    pin.updateFramePlacements(model, data)
    foot_fk = np.concatenate([data.oMf[fid].translation for fid in frame_ids])  # (6,)

    return q_ref, q_noisy, foot_fk

# ———————— 3) Example usage ————————
if __name__ == "__main__":
    # Example foot x-offsets and swing height
    foot_x = np.array([0.2, 0.25])
    z_swing = 0.05
    q_ref, q_err, foot_out = ik_fk_demo(foot_x, z_swing)

    print("IK angles    :", q_ref)
    print("Noisy angles :", q_err)
    print("FK foot pos  :", foot_out)
