#!/usr/bin/env python3
"""
amber_visualize_mjc.py  –  v-x only, 4-dof legs
"""

import argparse, time, numpy as np, mujoco, mujoco.viewer
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# paths / constants
# ──────────────────────────────────────────────────────────────────────
REF_DIR   = Path(__file__).resolve().parent / "references"
MODEL_XML = "Amber/amber_free.urdf"           # edit if you move it

BASE_Z      = 0.33        # standing height
DOF_AMBER   = 4           # q1L,q2L,q1R,q2R  (no passive joints)
PHASE_OFF   = np.array([0.0, 0.2])            # not used here but kept

# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--vx", type=float, help="forward speed to visualise (m/s)")
p.add_argument("--rate", type=float, default=0.25,
               help="sim-speed factor (default 0.25 = slow-mo)")
args = p.parse_args()

# ──────────────────────────────────────────────────────────────────────
# load reference library (vx only)
# ──────────────────────────────────────────────────────────────────────
vxs    = np.load(REF_DIR / "amber_vxs.npy")                    # (Nv,)
ts     = np.load(REF_DIR / "amber_reference_ts.npy")           # (N,)
qrefs  = np.load(REF_DIR / "amber_reference_qs.npy")           # (Nv,N,4)

Nv, N, _ = qrefs.shape
T        = ts[-1]

vx_des = args.vx if args.vx is not None else vxs[len(vxs)//2]
if vx_des <= vxs[0]:
    i0 = i1 = 0;               w = 1.0
elif vx_des >= vxs[-1]:
    i0 = i1 = Nv-1;            w = 1.0
else:
    i1 = np.searchsorted(vxs, vx_des)
    i0 = i1 - 1
    w  = (vx_des - vxs[i0]) / (vxs[i1]-vxs[i0])

print(f"Visualising vₓ = {vx_des:.2f}  (grid {i0}–{i1}, w={w:.2f})")

# ──────────────────────────────────────────────────────────────────────
# Mujoco model & viewer
# ──────────────────────────────────────────────────────────────────────
model = mujoco.MjModel.from_xml_path(MODEL_XML)
data  = mujoco.MjData(model)
# detect whether the model has a floating base
if model.nq == 7 + DOF_AMBER:          # 11 for Amber
    base_free = True
elif model.nq == DOF_AMBER:            # 4  for fixed base
    base_free = False
else:
    raise RuntimeError(f"Unexpected nq={model.nq}")
dt_phys  = model.opt.timestep
dt_vis   = 1/60.0
rate     = args.rate

# ---------------------------------------------------------------------
def quat_yaw(yaw):
    c, s = np.cos(0.5*yaw), np.sin(0.5*yaw)
    return np.array([c, 0.0, 0.0, s])       # w x y z

def sample_q(t_now):
    t = t_now % T
    k1 = np.searchsorted(ts, t)
    k0 = k1-1 if k1>0 else 0
    a  = (t-ts[k0])/(ts[k1]-ts[k0]) if k1>k0 else 0.0
    q0 = (1-w)*qrefs[i0,k0] + w*qrefs[i1,k0]
    q1 = (1-w)*qrefs[i0,k1] + w*qrefs[i1,k1]
    return (1-a)*q0 + a*q1                  # (4,)

# ──────────────────────────────────────────────────────────────────────
# simulate
# ──────────────────────────────────────────────────────────────────────
with mujoco.viewer.launch_passive(model, data) as viewer:
    t_sim = 0.0
    while viewer.is_running():
        start = time.time()

        q_full = sample_q(t_sim)             # (7,)  or (4,) if you regenerate
        q_leg  = q_full[:DOF_AMBER]          # keep first 4 actuated joints
        print(q_leg)
        # base + joints
        if base_free:
            data.qpos[:7]              = np.hstack((vx_des*t_sim, 0.0, BASE_Z, quat_yaw(0.0)))
            data.qpos[7:7+DOF_AMBER]   = q_leg
        else:
            # fixed base: only joints
            data.qpos[:DOF_AMBER]      = q_leg

        mujoco.mj_forward(model, data)
        viewer.sync()

        t_sim += rate * dt_phys
        time.sleep(max(0.0, dt_vis - (time.time() - start)))
