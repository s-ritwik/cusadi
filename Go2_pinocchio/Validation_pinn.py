#!/usr/bin/env python3
"""
Combined IK/FK Validation for Go2 Quadruped using MuJoCo and Pinocchio

This script:
1. Loads the MuJoCo XML (ground-truth) and the fixed URDF (with matching joint axes).
2. Defines a fixed reference set of foot positions.
3. Runs MuJoCo’s inverse kinematics (IK) to get q_mj (12×1), starting from zero.
4. Runs Pinocchio’s inverse kinematics (IK) to get q_pin (12×1), starting from the “standing” guess.
5. Runs MuJoCo’s forward kinematics (FK) on q_pin (treating q_pin as if it were a MuJoCo qpos) → foot positions.
6. Runs Pinocchio’s forward kinematics (FK) on q_mj → foot positions.
7. Prints:
   - The target foot positions.
   - The IK solutions (q_mj vs. q_pin).
   - The FK results (MuJoCo FK on q_mj, Pinocchio FK on q_pin).
   - The cross-validation FK results (MuJoCo FK on q_pin, Pinocchio FK on q_mj).
   - The per-foot Cartesian errors in each case.
"""

import mujoco
import numpy as np
import pinocchio as pin
from scipy.linalg import pinv

# ---------------------------
# 1. MuJoCo IK/FK functions
# ---------------------------

def mujoco_inverse_kinematics(target_xyz, foot_ids, model: mujoco.MjModel, data: mujoco.MjData,
                              q_init=None, tol=1e-6, max_iter=500, damping=1e-4):
    """
    Solve for qpos using damped least-squares IK on MuJoCo. 
    target_xyz: (4×3) array of desired foot positions in world frame.
    foot_ids: list of 4 MuJoCo site IDs.
    If q_init is not None, data.qpos[:] = q_init before iterating. 
    Returns a (12,) qpos vector [FL_hip, FL_thigh, FL_calf, FR_hip, …, RR_calf].
    """
    if q_init is not None:
        data.qpos[:] = q_init.copy()
    else:
        data.qpos[:] = np.zeros_like(data.qpos)

    mujoco.mj_forward(model, data)

    # Solve each foot one at a time (damped L-S)
    for idx, sid in enumerate(foot_ids):
        tgt = target_xyz[idx]
        for _ in range(max_iter):
            mujoco.mj_forward(model, data)
            cur = data.site_xpos[sid].copy()
            err = tgt - cur
            if np.linalg.norm(err) < tol:
                break
            Jpos = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, Jpos, None, sid)
            Jpinv = Jpos.T @ np.linalg.inv(Jpos @ Jpos.T + damping * np.eye(3))
            data.qpos[:] += Jpinv @ err
        if np.linalg.norm(err) >= tol:
            print(f"[MuJoCo IK WARN] site {sid} not converged, ‖err‖={np.linalg.norm(err):.2e}")

    mujoco.mj_forward(model, data)
    return data.qpos.copy()


def mujoco_forward_kinematics(qpos, foot_ids, model: mujoco.MjModel, data: mujoco.MjData):
    """
    Given a (12,) qpos array, run MuJoCo FK and return a (4×3) array of foot site positions.
    """
    data.qpos[:] = qpos.copy()
    mujoco.mj_forward(model, data)
    return np.vstack([data.site_xpos[sid].copy() for sid in foot_ids])


# -------------------------------------
# 2. Pinocchio IK/FK convenience funcs
# -------------------------------------

def inverse_kinematics_pinocchio(targets_xyz: np.ndarray,
                                  foot_frame_ids: list,
                                  model: pin.Model,
                                  data: pin.Data,
                                  q_init: np.ndarray,
                                  tol=1e-6,
                                  max_iter=100,
                                  damping=1e-4):
    """
    Solve for q (12×1) using damped least-squares IK in Pinocchio. 
    targets_xyz: (4×3) array of desired foot positions (in base frame).
    foot_frame_ids: list of 4 frame IDs for FL_foot, FR_foot, RL_foot, RR_foot.
    q_init: (12,) initial guess.
    Returns a (12,) q vector.
    """
    q = q_init.copy()

    for k, fid in enumerate(foot_frame_ids):
        tgt = targets_xyz[k]
        for _ in range(max_iter):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)

            p_cur = data.oMf[fid].translation
            err = tgt - p_cur
            if np.linalg.norm(err) < tol:
                break

            J6 = pin.computeFrameJacobian(model, data, q, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            J3 = J6[:3, :]  # position rows
            J_dls = J3.T @ np.linalg.inv(J3 @ J3.T + damping * np.eye(3))
            q += J_dls @ err

        if np.linalg.norm(err) >= tol:
            print(f"[Pinocchio IK WARN] Foot index {k} (frame {fid}) did not converge (‖err‖={np.linalg.norm(err):.3e} m)")

    return q


def pinocchio_forward_kinematics(q, foot_frame_ids: list, model: pin.Model, data: pin.Data):
    """
    Given a (12,) q, run Pinocchio FK and return a (4×3) array of foot positions (in base frame).
    """
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return np.vstack([data.oMf[fid].translation.copy() for fid in foot_frame_ids])


# ------------------
# 3. Main Validation
# ------------------

if __name__ == "__main__":
    # 3.1 Paths to models
    MJ_XML_PATH   = "go_2/go2_fixed.xml"
    URDF_PATH     = "Go2_pinocchio/go2_corrected_fixed3.urdf"

    # 3.2 Load MuJoCo model and data
    mj_model = mujoco.MjModel.from_xml_path(MJ_XML_PATH)
    mj_data  = mujoco.MjData(mj_model)

    # Collect site IDs for foot sites
    site_names   = ["FL_foot_site", "FR_foot_site", "RL_foot_site", "RR_foot_site"]
    foot_site_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, nm)
                     for nm in site_names]

    # 3.3 Load Pinocchio model and data
    pin_model = pin.buildModelFromUrdf(URDF_PATH)
    pin_data  = pin_model.createData()
    # Collect frame IDs for foot frames
    foot_frame_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    foot_frame_ids   = [pin_model.getFrameId(nm) for nm in foot_frame_names]

    # 3.4 Define a fixed reference foot positions (4×3)
    target_foot_positions = np.array([
        [ 0.17782152,  0.11044377, 0.12571125],   # FL
        [ 0.17782152, -0.11044377, 0.12571125],   # FR
        [-0.27051568,  0.11137226, 0.13496522],   # RL
        [-0.27051568, -0.11137226, 0.13496522],   # RR
    ])
    target_foot_positions= np.array([
    [ 0.180,  0.110, 0.126],   # FL
    [ 0.180, -0.110, 0.126],   # FR
    [-0.271,  0.111, 0.135],   # RL
    [-0.271, -0.111, 0.135],   # RR
    ]) 
    print("\n=== TARGET FOOT POSITIONS ===")
    for leg, pos in zip(["FL","FR","RL","RR"], target_foot_positions):
        print(f" {leg}: {pos}")

    # 3.5 Initial guess for Pinocchio IK: “neutral” standing pose
    q_init_pin = pin.neutral(pin_model)
    q_init_pin[:] = np.deg2rad([
        -5.7, 45.8, -86.0,   # FL (hip, thigh, calf)
         5.7, 45.8, -86.0,   # FR
        -5.7, 57.3, -86.0,   # RL
         5.7, 57.3, -86.0    # RR
    ])

    # 3.6 Initial guess for MuJoCo IK: **all zeros** (instead of “standing”)
    q_init_mj = np.zeros(12)

    # 3.7 Run MuJoCo IK → q_mj (now starts from zeros)
    q_mj = mujoco_inverse_kinematics(target_foot_positions, foot_site_ids,
                                     mj_model, mj_data,
                                     q_init=None,      # Pass None so it uses zeros internally
                                     tol=1e-6, max_iter=500, damping=1e-4)

    # 3.8 Run Pinocchio IK → q_pin (starts from the standing pose)
    q_pin = inverse_kinematics_pinocchio(target_foot_positions,
                                          foot_frame_ids,
                                          pin_model, pin_data,
                                          q_init=q_init_pin,
                                          tol=1e-6, max_iter=200, damping=1e-4)

    # 3.9 Print IK solutions (reshaped to 4×3 for each leg)
    print("\n=== IK SOLUTIONS ===")
    print("MuJoCo IK (q_mj) [rad], shape (4×3):")
    print(q_mj.reshape(4, 3))
    print("\nPinocchio IK (q_pin) [rad], shape (4×3):")
    print(q_pin.reshape(4, 3))

    # 3.10 Run MuJoCo FK on q_mj (sanity: should reproduce target)
    fk_mj_on_mj = mujoco_forward_kinematics(q_mj, foot_site_ids, mj_model, mj_data)

    # 3.11 Run Pinocchio FK on q_pin (sanity: should reproduce target)
    fk_pin_on_pin = pinocchio_forward_kinematics(q_pin, foot_frame_ids, pin_model, pin_data)

    # 3.12 Cross FK: MuJoCo FK on q_pin
    fk_mj_on_pin = mujoco_forward_kinematics(q_pin, foot_site_ids, mj_model, mj_data)

    # 3.13 Cross FK: Pinocchio FK on q_mj
    fk_pin_on_mj = pinocchio_forward_kinematics(q_mj, foot_frame_ids, pin_model, pin_data)

    # 3.14 Print forward-kinematics results
    print("\n=== FORWARD KINEMATICS RESULTS ===")

    # MuJoCo FK on its own IK
    print("\nMuJoCo FK on q_mj → foot pos (should be ≈ target):")
    for idx, leg in enumerate(["FL","FR","RL","RR"]):
        pos = fk_mj_on_mj[idx]
        err = np.linalg.norm(target_foot_positions[idx] - pos)
        print(f" {leg}: {pos}   (err = {err:.3e} m)")

    # Pinocchio FK on its own IK
    print("\nPinocchio FK on q_pin → foot pos (should be ≈ target):")
    for idx, leg in enumerate(["FL","FR","RL","RR"]):
        pos = fk_pin_on_pin[idx]
        err = np.linalg.norm(target_foot_positions[idx] - pos)
        print(f" {leg}: {pos}   (err = {err:.3e} m)")

    # MuJoCo FK on Pinocchio’s IK
    print("\nMuJoCo FK on q_pin → foot pos (cross-check):")
    for idx, leg in enumerate(["FL","FR","RL","RR"]):
        pos = fk_mj_on_pin[idx]
        err = np.linalg.norm(target_foot_positions[idx] - pos)
        print(f" {leg}: {pos}   (err = {err:.3e} m)")

    # Pinocchio FK on MuJoCo’s IK
    print("\nPinocchio FK on q_mj → foot pos (cross-check):")
    for idx, leg in enumerate(["FL","FR","RL","RR"]):
        pos = fk_pin_on_mj[idx]
        err = np.linalg.norm(target_foot_positions[idx] - pos)
        print(f" {leg}: {pos}   (err = {err:.3e} m)")

    # 3.15 Summary of errors
    print("\n=== SUMMARY OF ERRORS ===")
    print("MuJoCo IK → MuJoCo FK max error (m):", np.max(np.linalg.norm(target_foot_positions - fk_mj_on_mj, axis=1)))
    print("Pinocchio IK → Pinocchio FK max error (m):", np.max(np.linalg.norm(target_foot_positions - fk_pin_on_pin, axis=1)))
    print("Pinocchio IK → MuJoCo FK max error (m):",  np.max(np.linalg.norm(target_foot_positions - fk_mj_on_pin, axis=1)))
    print("MuJoCo IK → Pinocchio FK max error (m):",  np.max(np.linalg.norm(target_foot_positions - fk_pin_on_mj, axis=1)))
