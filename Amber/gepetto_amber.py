#!/usr/bin/env python3
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer
import numpy as np
import time
import subprocess
import sys
from pathlib import Path
import argparse

def launch_gepetto_gui():
    try:
        subprocess.Popen(["gepetto-gui"])
        print("[INFO] Started gepetto-gui.")
    except FileNotFoundError:
        print("[ERROR] gepetto-gui executable not found. Is it installed?")
    except Exception as e:
        print(f"[ERROR] Could not start gepetto-gui: {e}")

def yaw2quat(theta: float) -> np.ndarray:
    cy = np.cos(theta * 0.5)
    sy = np.sin(theta * 0.5)
    # Pinocchio expects [x, y, z, w]
    return np.array([0.0, 0.0, sy, cy])

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Visualize Amber biped gait library in Gepetto-GUI"
    )
    p.add_argument(
        "--vx", type=float, default=None,
        help="Forward speed (m/s) to visualize (must match one of the precomputed vxs)"
    )
    p.add_argument(
        "--rate", type=float, default=1.0,
        help="Playback speed multiplier (1.0 = real time)"
    )
    args = p.parse_args()

    # 1) Launch GUI
    launch_gepetto_gui()
    time.sleep(2.0)

    # 2) Load gait library
    ref_dir = Path("Amber/references")
    vxs = np.load(ref_dir / "amber_vxs.npy")
    ts  = np.load(ref_dir / "amber_reference_ts.npy")
    q_refs = np.load(ref_dir / "amber_reference_qs.npy")         # shape (n_vx, N, 4)
    # foot_refs = np.load(ref_dir / "amber_reference_foot_refs.npy")  # if desired

    # choose the speed index
    if args.vx is None:
        ix = 0
    else:
        ix = int(np.argmin(np.abs(vxs - args.vx)))
    vx = float(vxs[ix])
    print(f"[INFO] Visualizing vx = {vx:.3f} m/s (index {ix})")

    # 3) Build Pinocchio model with a floating base
    urdf_path = Path("Amber/amber.urdf")
    mesh_dirs = [str(urdf_path.parent)]
    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        str(urdf_path), mesh_dirs, pin.JointModelFreeFlyer()
    )
    data = model.createData()

    # 4) Init visualizer
    viz = GepettoVisualizer(model, collision_model, visual_model)
    viz.initViewer(loadModel=True)
    viz.loadViewerModel("amber")

    # 5) Precompute base height so that feet sit on z=0 plane
    #    assume q_joints = zeros gives default foot placement
    q0 = np.zeros(model.nq)
    # default quat = identity
    q0[3:7] = np.array([0,0,0,1])
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)
    # foot frame names must match your URDF
    foot_names = ["left_toe", "right_toe"]
    fids = [model.getFrameId(n) for n in foot_names]
    foot_zs = [data.oMf[fid].translation[2] for fid in fids]
    z_base = -min(foot_zs)
    print(f"[INFO] Setting base height to {z_base:.3f} so feet start at z=0")

    # 6) Prepare time & transforms
    N = ts.shape[0]
    dt = 1.0/60.0
    x_array     = vx * ts
    y_array     = np.zeros_like(ts)
    theta_array = np.zeros_like(ts)

    # 7) Animation loop
    try:
        for i in range(N):
            # Build full q vector: [x, y, z, quat (x,y,z,w), joint1..joint4]
            q = np.zeros(model.nq)
            # print("  model.nq        =", model.nq)
            # print("  q_refs.shape    =", q_refs.shape)
            # print("  q[7:].shape     =", q[7:].shape)
            q[0:3] = [x_array[i], y_array[i], z_base]
            q[3:7] = yaw2quat(theta_array[i])
            q[7:] = q_refs[ix, i, :]  # the 4 joint angles

            # display
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            viz.display(q)

            # sleep to maintain 60 Hz
            time.sleep(max(0.0, dt/args.rate))
    except KeyboardInterrupt:
        print("\n[INFO] Animation stopped by user.")
        sys.exit(0)
