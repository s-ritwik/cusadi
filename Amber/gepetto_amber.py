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
        "--vx", type=float, default=0.2,
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
    # print(ts)
    q_refs = np.load(ref_dir / "amber_reference_qs.npy")         # shape (n_vx, N, 4)
    print("vxs",q_refs.shape)

    # foot_refs = np.load(ref_dir / "amber_reference_foot_refs.npy")  # if desired
    joint_dim = q_refs.shape[-1]   # = 7 in your case
    print("Joint_dim",joint_dim)
    # choose the speed index
    if args.vx is None:
        ix = 0
    else:
        ix = int(np.argmin(np.abs(vxs - args.vx)))
    vx = float(vxs[ix])
    print(f"[INFO] Visualizing vx = {vx:.3f} m/s (index {ix})")

    # 3) Build Pinocchio model with a floating base
    urdf_path = Path("Amber/amber_free.urdf")
    mesh_dirs = [str(urdf_path.parent)]
    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        str(urdf_path), mesh_dirs, pin.JointModelFreeFlyer()
    )
    data = model.createData()
    name_to_idx = { model.names[jid]: model.joints[jid].idx_q
                for jid in range(model.njoints) }

    actuated_names = ["q1_left","q2_left","q1_right","q2_right"]
    
    actuated_idxs  = [ name_to_idx[n] for n in actuated_names ]
    print(actuated_idxs)
    print(" model.nq =", model.nq)
    for jid in range(model.njoints):
        print(f"  Joint {jid:2d}: name = {model.names[jid]:30s}",
            f"idx_q = {model.joints[jid].idx_q:2d},  nq = {model.joints[jid].nq}")
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
    z_base = -min(foot_zs)-0.1
    print(f"[INFO] Setting base height to {z_base:.3f} so feet start at z=0")

    # 6) Prepare time & transforms
    N = ts.shape[0]
    ts_val=ts[-1]
    dt = ts_val/N
    x_array     = vx * ts
    y_array     = np.zeros_like(ts)
    theta_array = np.zeros_like(ts)
    T        = ts.max()        # duration of one cycle
    x_offset = 0.0             # how far we’ve walked so far
    phase_offset = np.array([0.0, T/2.0])
    # 7) Animation loop
    while True:
        try:
            for i in range(N):
                # Build full q vector: [x, y, z, quat (x,y,z,w), joint1..joint4]
                q = np.zeros(model.nq)
                # print("  model.nq        =", model.nq)
                # print("  q_refs.shape    =", q_refs.shape)
                # print("  q[7:].shape     =", q[7:].shape)
                x_curr = x_offset + vx*ts[i]
                q[0:3] = [x_curr, y_array[i], z_base]
                q[3:7] = yaw2quat(theta_array[i])
                # actuated = q_refs[ix, i, :]            # shape (4,)
                # print("actuated shape",actuated.shape[0])
                # print(q_refs[ix,i,:])
                # q[actuated_idxs] = q_refs[ix, i, :]              # print(" actuated joints →", q[7:11])
                q_left = q_refs[ix, i, :2]

                #  • right leg uses time shifted by half a cycle
                t_phase = (ts[i] + phase_offset[1]) % T
                # find the nearest index into ts
                idx_phase = int(np.argmin(np.abs(ts - t_phase)))
                q_right = q_refs[ix, idx_phase, 2:4]
                q[actuated_idxs[0:2]] = q_left
                q[actuated_idxs[2:4]] = q_right
                # display
                pin.forwardKinematics(model, data, q)
                pin.updateFramePlacements(model, data)
                viz.display(q)
                # print("in loo[]")
                # sleep to maintain 60 Hz
                time.sleep(max(0.0, dt/args.rate))
            x_offset += vx * T    
        except KeyboardInterrupt:
            print("\n[INFO] Animation stopped by user.")
            sys.exit(0)
