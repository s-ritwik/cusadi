import pinocchio as pin
import numpy as np
import time
import sys
import subprocess

# MuJoCo imports
import mujoco
from mujoco.viewer import launch_passive

def launch_gepetto_gui():
    """(Optional) stub for launching gepetto-gui in a separate terminal."""
    try:
        if sys.platform.startswith('linux'):
            subprocess.Popen([
                "gnome-terminal",
                "--", "bash", "-c", "gepetto-gui & exec bash"
            ])
            print("[INFO] (Stub) Launched gepetto-gui in a new terminal.")
        else:
            print("[WARN] Automatic gepetto-gui launch not implemented for this OS.")
    except Exception as e:
        print(f"[ERROR] Failed to launch gepetto-gui: {e}")

if __name__ == "__main__":
    # (Optional) still launching Gepetto if you want to compare side-by-side:
    # launch_gepetto_gui()
    time.sleep(2.0)

    # Desired base velocities (m/s, rad/s)
    v_x = 0.2
    v_y = 0.2
    w_z = 0.2

    def yaw2quat(theta):
        """Convert yaw angle (about Z) → MuJoCo quaternion ordering [x,y,z,w]."""
        cy = np.cos(theta * 0.5)
        sy = np.sin(theta * 0.5)
        return np.array([0.0, 0.0, sy, cy])

    # 1) Load your precomputed reference trajectories (numpy arrays)
    v_xs      = np.load("references/go2_vxs.npy")      # shape: ( ... )
    v_ys      = np.load("references/go2_vys.npy")
    w_zs      = np.load("references/go2_wzs.npy")
    ts        = np.load("references/go2_reference_ts.npy")
    q_refs    = np.load("references/go2_reference_qs.npy")        # shape: (nx, ny, nw, nt, 12)
    foot_refs = np.load("references/go2_reference_foot_refs.npy") # (not used here, but loaded for completeness)

    # 2) Build Pinocchio model from URDF (free-flyer + 12 joints)
    URDF_PATH = "/home/s-ritwik/src/cusadi/Go2_pinocchio/old/go2_corrected_fixed3.urdf"
    MESH_DIRS = ["Go2_pinocchio"]  # where your .stl/.obj meshes live

    try:
        model, collision_model, visual_model = pin.buildModelsFromUrdf(
            URDF_PATH,
            MESH_DIRS,
            pin.JointModelFreeFlyer()   # → adds 7 DOF floating base at root
        )
    except Exception as e:
        print(f"[ERROR] Could not load URDF or meshes: {e}")
        sys.exit(1)
    data = model.createData()  # For Pinocchio’s FK/IK

    # 3) Tell MuJoCo to load exactly that same URDF (no separate XML)
    #    MuJoCo’s Python bindings will run its URDF→MJCF converter under the hood.
    try:
        mj_model = mujoco.MjModel.from_xml_path(URDF_PATH)
    except Exception as e:
        print(f"[ERROR] MuJoCo failed to load URDF '{URDF_PATH}': {e}")
        sys.exit(1)
    mj_data = mujoco.MjData(mj_model)

    # How many qpos does MuJoCo expect?
    mj_nq = mj_model.nq
    # For a free-flyer + 12 joints URDF, mujuco.nq should be 7 + 12 = 19.
    # But if your URDF actually fixed the base, then mj_nq might be 12. We'll branch below.

    # 4) Launch a passive (non-blocking) MuJoCo viewer
    viewer = launch_passive(mj_model, mj_data)

    # 5) Precompute index‐lookups for interpolation (same as your original code)
    vx_low   = np.where(v_xs <= v_x)[0][-1]
    vx_high  = np.argmax(v_xs >= v_x)
    vy_low   = np.where(v_ys <= v_y)[0][-1]
    vy_high  = np.argmax(v_ys >= v_y)
    wz_low   = np.where(w_zs <= w_z)[0][-1]
    wz_high  = np.argmax(w_zs >= w_z)

    # The gait period T (max timestamp)
    T = ts.max()
    phase_offset = np.array([0.0, T/2.0, T/2.0, 0.0])  # 4-leg phases
    t = 0.0
    dt = 1.0 / 60.0   # drive loop at ~60 Hz
    sim_rate = 0.2    # scaling factor for input velocity

    while True:
        t0 = time.time()
        t  += sim_rate * dt
        theta = w_z * t

        # 6) Compute free-flyer XY position under constant yaw‐rate w_z
        if abs(w_z) > 1e-3:
            x = (v_x * np.sin(w_z * t) + v_y * (np.cos(w_z * t) - 1)) / w_z
            y = (v_x * (1 - np.cos(w_z * t)) + v_y * np.sin(w_z * t)) / w_z
        else:
            x = v_x * t
            y = v_y * t

        # 7) Build full Pinocchio state q_pin (length = model.nq = 19)
        q_pin = np.zeros(model.nq)  # model.nq == 19
        z_base = -0.1               # fixed z height for base
        quat = yaw2quat(theta)      # [x, y, z, w]
        q_pin[0:3]  = np.array([x, y, z_base])
        q_pin[3:7]  = quat

        # 8) Interpolate leg joint angles exactly the same as before
        vx_inds  = np.array([vx_low] * 8 + [vx_high] * 8)
        vy_inds  = np.tile(np.array([vy_low] * 4 + [vy_high] * 4), 2)
        wz_inds  = np.tile(np.array([wz_low] * 2 + [wz_high] * 2), 4)

        q_joints = np.zeros(12)
        for foot_id in range(4):
            t_phase = (t + phase_offset[foot_id]) % T
            ts_low  = np.where(ts <= t_phase)[0][-1]
            ts_high = np.argmax(ts >= t_phase)

            t_inds = np.tile(np.array([ts_low, ts_high]), 8)

            # Gather the four corner data points for interpolation
            vecs = np.stack([
                v_xs[vx_inds],
                v_ys[vy_inds],
                w_zs[wz_inds],
                ts[t_inds]
            ], axis=1)

            des_vec = np.array([v_x, v_y, w_z, t_phase])
            dist = np.linalg.norm(vecs - des_vec, axis=1) + 1e-4
            weights = (1.0 / dist)
            weights /= np.sum(weights)

            start = 3 * foot_id
            end   = 3 * (foot_id + 1)
            q_candidates = q_refs[
                vx_inds,
                vy_inds,
                wz_inds,
                t_inds,
                start:end
            ]
            q_joints[start:end] = np.sum(weights[:, None] * q_candidates, axis=0)

        # Fill in the 12 leg joints into q_pin
        q_pin[7:] = q_joints  # indices 7..18 (12 elements)

        # —————— Push q_pin into MuJoCo ——————
        if mj_nq == model.nq:
            # Case A: MuJoCo expects exactly 19 qpos (free-flyer + 12 joints)
            mj_data.qpos[:] = q_pin
        elif mj_nq == (model.nq - 7):
            # Case B: MuJoCo only has 12 actuated joints (base is implicitly fixed)
            #    So drop the first 7 entries of q_pin, keep only indices 7..18 → 12 joint angles
            mj_data.qpos[:] = q_pin[7:]
        else:
            raise RuntimeError(
                f"MuJoCo n_qpos = {mj_nq}, but Pinocchio model.nq = {model.nq}."
                " Cannot decide how to assign qpos."
            )

        # 9) Since we’re not simulating dynamics, just run forward kinematics:
        mujoco.mj_forward(mj_model, mj_data)

        # 10) Tell the passive viewer to redraw at this new qpos:
        viewer.sync()

        # Keep roughly 60 Hz
        elapsed = time.time() - t0
        time.sleep(max(0.0, dt - elapsed))
