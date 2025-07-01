#!/usr/bin/env python3
"""
CPU-side trajectory library generator for the Amber biped.
Exactly mirrors go2_casadi_main_cpu.py but for:
  - 2 feet  → vectors length 6
  - 4 joints→ vectors length 4
"""
import os
import numpy as np, time, casadi as ca, pinocchio as pin
print(ca.__version__)

# ----------------------------------------------------------------------
# 1) Load the compiled IK helper
# ----------------------------------------------------------------------
reference_step = ca.Function.load("amber_reference_step.casadi")

# ----------------------------------------------------------------------
# 2) Pinocchio model & constants
# ----------------------------------------------------------------------
URDF          = "Amber/amber.urdf"
FOOT_FRAMES   = ["left_toe", "right_toe"]
N_LEGS        = len(FOOT_FRAMES)          # 2
DAMPING_PIN   = 1e-4
IK_ITERS_PIN  = 100
IK_TOL_PIN    = 1e-6

model_pin = pin.buildModelFromUrdf(URDF)
data_pin  = model_pin.createData()
frame_ids = [model_pin.getFrameId(f) for f in FOOT_FRAMES]
DOF           = model_pin.nq                # 4

q_init          = np.zeros(DOF)           # neutral pose = all zeros
q_init_dm       = ca.DM(q_init.reshape(DOF,1))

# ----------------------------------------------------------------------
# 3) Bézier helper (unchanged)
# ----------------------------------------------------------------------
def cubic_bezier(z0, z1, t):
    t = ca.fmax(0, ca.fmin(1, t))
    return z0 + (z1-z0)*(t**3 + 3*t**2*(1-t))

# ----------------------------------------------------------------------
# 4) Reference-generator  (all dimensions driven by N_LEGS / DOF)
# ----------------------------------------------------------------------
def generate_reference(vx,
                       foot_ids, default_foot_pos, swing_h, T, N):

    def rot_z(a):   # DM 3×3
        c,s = ca.cos(a), ca.sin(a)
        return ca.horzcat(
            ca.vertcat(c, s, 0),
            ca.vertcat(-s, c, 0),
            ca.vertcat(0, 0, 1))

    ts_np  = np.linspace(0, T, N)
    ts     = ca.DM(ts_np)
    phase  = ca.DM(np.clip(ts_np/T*2-0.5, 0, 1))     # 0→1 ramp
    mask   = (phase.full().flatten() <= 0.5)

    z0 = cubic_bezier(0, swing_h, 2*phase)
    z1 = cubic_bezier(swing_h, 0, 2*phase-1)
    z  = ca.if_else(ca.DM(mask), z0, z1)

    theta = ca.DM.zeros(ts.shape)
    x     = vx * ts
    y     = ca.DM.zeros(ts.shape)           # no lateral motion
    # default foot matrix  (2×3)
    foot_def = ca.horzcat(*[ca.DM(default_foot_pos[f]).reshape((3, 1))
                            for f in foot_ids])
    # Pre-compute start/end world foot targets
    p0_com = ca.vertcat(x[0],   y[0],   0)
    p1_com = ca.vertcat(x[-1],  y[-1],  0)
    R0, R1 = rot_z(theta[0]), rot_z(theta[-1])
    p0_ft  = ca.repmat(p0_com.T, N_LEGS, 1) + (R0 @ foot_def).T
    p1_ft  = ca.repmat(p1_com.T, N_LEGS, 1) + (R1 @ foot_def).T

    # Stack world-frame foot trajectories   (6×N)
    foot_w_stack = ca.DM.zeros(N_LEGS*3, N)
    for i in range(N):
        ft_i = cubic_bezier(p0_ft, p1_ft, phase[i])
        foot_w_stack[:,i] = ca.reshape(ft_i, N_LEGS*3, 1)

    # Output arrays
    q_ref   = ca.DM.zeros(N, DOF)
    foot_b  = ca.DM.zeros(N, N_LEGS*3)

    for i in range(N):
        qi, fi = reference_step(phase[i],
                                foot_w_stack[:,i],
                                x[i], 0.0, 0.0, z[i],
                                q_init_dm)
        q_ref[i,:]  = ca.reshape(qi, 1, DOF)
        foot_b[i,:] = ca.reshape(fi, 1, N_LEGS*3)

    return ts_np, q_ref.toarray(), foot_b.toarray().reshape(N,N_LEGS,3)

# ----------------------------------------------------------------------
# 5) Library-builder  (unchanged logic, new dimensions)
# ----------------------------------------------------------------------
def generate_gait_library(vxs,
                          swing_h=0.06, T=0.4, N=100):

    # default foot world positions from neutral pose
    pin.forwardKinematics(model_pin, data_pin, q_init)
    pin.updateFramePlacements(model_pin, data_pin)
    default_foot = {fid: data_pin.oMf[fid].translation.copy()
                    for fid in frame_ids}

    q_refs = np.zeros((len(vxs), N, DOF))
    foot_refs = np.zeros((len(vxs), N, N_LEGS, 3))
    ts_out = None

    for ix, vx in enumerate(vxs):
        print(f"Generate vx = {vx:.2f} m/s")
        ts_out, q, f = generate_reference(
            vx, frame_ids, default_foot,
            swing_h, T, N)
        q_refs[ix,:,:]      = q
        foot_refs[ix,:,:,:] = f

    # Save to disk
    os.makedirs("Amber/references", exist_ok=True)
    np.save("Amber/references/amber_vxs.npy", vxs)
    # np.save("references/amber_vys.npy", vys)
    # np.save("references/amber_wzs.npy", wzs)
    np.save("Amber/references/amber_reference_ts.npy", ts_out)
    np.save("Amber/references/amber_reference_qs.npy", q_refs)
    np.save("Amber/references/amber_reference_foot_refs.npy", foot_refs)

    return ts_out, q_refs, foot_refs

# ----------------------------------------------------------------------
# 6) Stand-alone call
# ----------------------------------------------------------------------
if __name__ == "__main__":
    vxs = np.linspace(-1.0, 1.2, 100)
    # vys = np.linspace(-0.4, 0.4, 5)
    # wzs = np.linspace(-0.5, 0.5, 5)
    ts, qs, fs = generate_gait_library(vxs)
    print("Amber gait library generated.")
