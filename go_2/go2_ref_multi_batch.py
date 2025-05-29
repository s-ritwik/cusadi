# go2_ref_gen_batch.py
import os, time
import numpy as np
import torch
import casadi
import mujoco
from src import *

# ── 1) USER GRID & PARAMS ─────────────────────────────────────────────────
v_xs = np.linspace(-1.0, 1.5, 11)    # original grid
v_ys = np.linspace(-0.75, 0.75, 7)
w_zs = np.linspace(-0.5, 0.5, 5)
swing_h = 0.08
T       = 0.4
N       = 10

# grid sizes and batch
nx, ny, nw = v_xs.size, v_ys.size, w_zs.size
BATCH = nx * ny * nw

# ── 2) LOAD IK KERNEL ──────────────────────────────────────────────────────

fn_cas = casadi.Function.load(os.path.join(CUSADI_FUNCTION_DIR,
                                           'fn_IK_go2.casadi'))
fn_gpu = CusadiFunction(fn_cas, BATCH)
# ── 3) FLATTEN GRID OF COMMANDS ────────────────────────────────────────────
VX, VY, WZ = np.meshgrid(v_xs, v_ys, w_zs, indexing='ij')
vx_flat = VX.reshape(-1)
vy_flat = VY.reshape(-1)
wz_flat = WZ.reshape(-1)

# ── 4) MuJoCo default foot positions ───────────────────────────────────────
model = mujoco.MjModel.from_xml_path("go_2/go2_fixed.xml")
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
foot_names = ["FL_foot_site","FR_foot_site","RL_foot_site","RR_foot_site"]
foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in foot_names]
default_foot_pos = np.stack([data.site_xpos[i] for i in foot_ids], axis=0)  # (4,3)

# ── 5) PREP STORAGE ─────────────────────────────────────────────────────────
ts = np.linspace(0, T, N)            # (N,)
q_flat = np.zeros((BATCH, N, 12))
foot_flat = np.zeros((BATCH, N, 4, 3))

# safe rotate batch

def rotate_batch(pts, thetas):
    B = thetas.shape[0]
    R = np.stack([
        [np.cos(thetas), -np.sin(thetas), np.zeros(B)],
        [np.sin(thetas),  np.cos(thetas), np.zeros(B)],
        [np.zeros(B),     np.zeros(B),    np.ones(B)]
    ], axis=-1).reshape(B,3,3)
    P = np.broadcast_to(default_foot_pos[None], (B,4,3))
    return (R[:,None] @ P[...,None]).squeeze(-1)

# compute endpoints at t=0 and t=T safely
den = wz_flat
mask = np.abs(den) > 1e-8
# allocate
xT = np.empty_like(vx_flat)
yT = np.empty_like(vy_flat)
# non-zero w_z
xT[mask] = (vx_flat[mask]*np.sin(den[mask]*T) + vy_flat[mask]*(np.cos(den[mask]*T)-1)) / den[mask]
yT[mask] = (vx_flat[mask]*(1-np.cos(den[mask]*T)) + vy_flat[mask]*np.sin(den[mask]*T)) / den[mask]
# zero w_z
xT[~mask] = vx_flat[~mask]*T
yT[~mask] = vy_flat[~mask]*T

p0 = np.zeros((BATCH,3))             # (B,3)
pT = np.stack((xT, yT, wz_flat*T), axis=1)  # (B,3)

foot0 = p0[:,None,:] + rotate_batch(default_foot_pos, p0[:,2])
foot1 = pT[:,None,:] + rotate_batch(default_foot_pos, pT[:,2])

# ── 6) VECTORIZED TIME‐STEPPING ───────────────────────────────────────────
for i, t in enumerate(ts):
    # phase and lift
    phase = np.clip((t/T - 0.25)*2, 0, 1)
    zb = ((2*phase)**3 + 3*(2*phase)**2*(1-2*phase)) if phase <= 0.5 else ((2*phase-1)**3 + 3*(2*phase-1)**2*(2-(2*phase-1)))
    z = zb * swing_h if phase <= 0.5 else 0

    # COM positions safely
    den = wz_flat
    mask = np.abs(den) > 1e-8
    xt = np.empty_like(vx_flat)
    yt = np.empty_like(vy_flat)
    xt[mask] = (vx_flat[mask]*np.sin(den[mask]*t) + vy_flat[mask]*(np.cos(den[mask]*t)-1)) / den[mask]
    yt[mask] = (vx_flat[mask]*(1-np.cos(den[mask]*t)) + vy_flat[mask]*np.sin(den[mask]*t)) / den[mask]
    xt[~mask] = vx_flat[~mask]*t
    yt[~mask] = vy_flat[~mask]*t
    th = wz_flat * t
    p_com = np.stack((xt, yt, th), axis=1)

    # interpolate foot global pos and lift
    fpw = foot0*(1-phase) + foot1*phase        # (B,4,3)
    Rts = rotate_batch(fpw - p_com[:,None,:], -th)
    Rts[...,2] += z

    foot_flat[:,i,:,:] = Rts

    # parallel IK
    inp = torch.from_numpy(Rts.reshape(BATCH,12).T).to(device='cuda', dtype=torch.double)
    torch.cuda.synchronize()
    fn_gpu.evaluate([inp])
    torch.cuda.synchronize()
    Q = fn_gpu.outputs_sparse[0].cpu().numpy()

    q_flat[:,i,:] = Q
    if i % 10 == 0:
        print(f"Step {i+1}/{N}")

# ── 7) RESTORE GRID SHAPE & ROTATE TO STANCE→SWING ─────────────────────────
q_refs = q_flat.reshape(nx, ny, nw, N, 12)
foot_refs = foot_flat.reshape(nx, ny, nw, N, 4, 3)
# rotate back
ind75 = int(ts.size * 0.75)
q_refs = np.concatenate((q_refs[..., ind75:, :], q_refs[..., :ind75, :]), axis=-2)
foot_refs = np.concatenate((foot_refs[..., ind75:, :, :], foot_refs[..., :ind75, :, :]), axis=-3)

# ── 8) SAVE EVERYTHING ─────────────────────────────────────────────────────

os.makedirs("go_2/references", exist_ok=True)
np.save("go_2/references/go2_vxs.npy", v_xs)
np.save("go_2/references/go2_vys.npy", v_ys)
np.save("go_2/references/go2_wzs.npy", w_zs)
np.save("go_2/references/go2_reference_ts.npy", ts)
np.save("go_2/references/go2_reference_qs.npy", q_refs)
np.save("go_2/references/go2_reference_foot_refs.npy", foot_refs)

print("Saved reference library for grid ", v_xs.shape, v_ys.shape, w_zs.shape, "→", q_refs.shape)