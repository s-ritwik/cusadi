import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer
import numpy as np
import time
import sys
# launch gui
import subprocess
import os

# Launch gepetto-gui in a new terminal
def launch_gepetto_gui():
    try:
        # Detect the shell and terminal
        if sys.platform.startswith('linux'):
            # Example with gnome-terminal
            subprocess.Popen(["gnome-terminal", "--", "bash", "-c", "gepetto-gui & exec bash"])
            # Alternative: use xterm if gnome-terminal is not installed
            # subprocess.Popen(["xterm", "-e", "gepetto-gui &"])
            print("[INFO] Launched gepetto-gui in a new terminal.")
        else:
            print("[WARN] Automatic gepetto-gui launch not implemented for this OS.")
    except Exception as e:
        print(f"[ERROR] Failed to launch gepetto-gui: {e}")

# Call this function at start
launch_gepetto_gui()

# Optional: wait a few seconds to let gepetto-gui initialize
time.sleep(2.0)

v_x = 0.2
v_y = 0.2
w_z = 0.2

def yaw2quat(theta):
    cy = np.cos(theta * 0.5)
    sy = np.sin(theta * 0.5)
    # Return in MUJOCO ORDER: [x, y, z, w]
    return np.array([0.0, 0.0, sy, cy])

if __name__ == "__main__":
    # Load saved reference data
    v_xs     = np.load("references/go2_vxs.npy")
    v_ys     = np.load("references/go2_vys.npy")
    w_zs     = np.load("references/go2_wzs.npy")
    ts       = np.load("references/go2_reference_ts.npy")
    q_refs   = np.load("references/go2_reference_qs.npy")         
    foot_refs = np.load("references/go2_reference_foot_refs.npy")  

    # Paths
    URDF_PATH = "Go2_pinocchio/go2_corrected_fixed3.urdf"
    MESH_DIRS = ["Go2_pinocchio"]   # adjust as needed to match your mesh location

    # Build model with visual + collision geometries
    try:
        model, collision_model, visual_model = pin.buildModelsFromUrdf(
            URDF_PATH, MESH_DIRS, pin.JointModelFreeFlyer())
    except Exception as e:
        print(f"[ERROR] Could not load URDF or meshes: {e}")
        sys.exit(1)
    data = model.createData()

    # Initialize GepettoVisualizer
    viz = None
    try:
        viz = GepettoVisualizer(model, collision_model, visual_model)
        viz.initViewer(loadModel=True)
        viz.loadViewerModel("go2")   # arbitrary name for the viewer
        has_viz = True
    except Exception as e:
        print(f"[WARN] GepettoVisualizer failed. Continuing without visualization:\n    {e}")
        has_viz = False

    # Precompute interpolation indices
    vx_low  = np.where(v_xs <= v_x)[0][-1]
    vx_high = np.argmax(v_xs >= v_x)
    vy_low  = np.where(v_ys <= v_y)[0][-1]
    vy_high = np.argmax(v_ys >= v_y)
    wz_low  = np.where(w_zs <= w_z)[0][-1]
    wz_high = np.argmax(w_zs >= w_z)

    T = ts.max()
    phase_offset = np.array([0.0, T/2.0, T/2.0, 0.0])  
    t = 0.0
    dt = 1.0 / 60.0     
    sim_rate = 0.2      

    while True:
        t0 = time.time()
        t += sim_rate * dt
        theta = w_z * t

        # Floating base XY motion
        if abs(w_z) > 1e-3:
            x = (v_x * np.sin(w_z * t) + v_y * (np.cos(w_z * t) - 1)) / w_z
            y = (v_x * (1 - np.cos(w_z * t)) + v_y * np.sin(w_z * t)) / w_z
        else:
            x = v_x * t
            y = v_y * t

        # Full q_pinocchio vector
        q_pin = np.zeros(model.nq)
        z_base = -.1
        quat = yaw2quat(theta)  
        q_pin[0:3] = np.array([x, y, z_base])
        q_pin[3:7] = quat

        # Interpolate joints
        vx_inds = np.array([vx_low] * 8 + [vx_high] * 8)
        vy_inds = np.tile(np.array([vy_low] * 4 + [vy_high] * 4), 2)
        wz_inds = np.tile(np.array([wz_low] * 2 + [wz_high] * 2), 4)

        q_joints = np.zeros(12)
        for foot_id in range(4):
            t_phase = (t + phase_offset[foot_id]) % T
            ts_low = np.where(ts <= t_phase)[0][-1]
            ts_high = np.argmax(ts >= t_phase)

            t_inds = np.tile(np.array([ts_low, ts_high]), 8)

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

        # Final q_pin
        q_pin[7:] = q_joints

        # Display in Gepetto
        if has_viz:
            pin.forwardKinematics(model, data, q_pin)
            pin.updateFramePlacements(model, data)
            try:
                viz.display(q_pin)
            except Exception as e:
                print(f"[WARN] Failed to viz.display(), disabling viz: {e}")
                has_viz = False

        # Maintain 60 Hz loop
        elapsed = time.time() - t0
        time.sleep(max(0.0, dt - elapsed))
