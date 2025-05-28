# go2_reference_gen_batch.py
import mujoco, numpy as np, torch, casadi, time, os
from src import *

# ── 1) USER PARAMETERS ────────────────────────────────────────────────────
BATCH      = 100   # how many robots in parallel
swing_h    = 0.08    # swing height (m)
T          = 0.4     # gait cycle duration (s)
N          = 100     # # of time steps
# velocity ranges (same as original)
vx_min, vx_max = -1.0, 1.5
vy_min, vy_max = -0.75, 0.75
wz_min, wz_max = -0.5, 0.5

# ── 2) LOAD IK KERNEL ────────────────────────────────────────────────────

fn_cas = casadi.Function.load(os.path.join(CUSADI_FUNCTION_DIR,
                                           'fn_IK_go2.casadi'))
fn_gpu = CusadiFunction(fn_cas, BATCH)
# ── 3) SAMPLE COMMANDS ────────────────────────────────────────────────────
# one (vx,vy,wz) per “robot”
v_xs = np.random.uniform(vx_min, vx_max, size=BATCH)
v_ys = np.random.uniform(vy_min, vy_max, size=BATCH)
w_zs = np.random.uniform(wz_min, wz_max, size=BATCH)

# ── 4) MuJoCo default foot positions ───────────────────────────────────────
model = mujoco.MjModel.from_xml_path("go_2/go2_fixed.xml")
data  = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
# site IDs for the four feet
foot_names = ["FL_foot_site","FR_foot_site","RL_foot_site","RR_foot_site"]
foot_ids   = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
              for name in foot_names]
# default positions in body frame (4×3)
default_foot_pos = np.stack([data.site_xpos[i] for i in foot_ids], axis=0)

# ── 5) PREPARE STORAGE ─────────────────────────────────────────────────────
ts         = np.linspace(0, T, N)                 # (N,)
q_refs     = np.zeros((BATCH, N, 12), dtype=np.float64)
foot_refs  = np.zeros((BATCH, N, 4, 3), dtype=np.float64)

# Precompute p_com at t=0 and t=T for each robot (BATCH×3)
p_com_0 = np.stack((v_xs*0, v_ys*0, w_zs*0), axis=1)  # always [0,0,0]
# at t=T:
if abs(w_zs).any():
    xT = (v_xs*np.sin(w_zs*T)+v_ys*(np.cos(w_zs*T)-1))/w_zs
    yT = (v_xs*(1-np.cos(w_zs*T))+v_ys*np.sin(w_zs*T))/w_zs
else:
    xT = v_xs*T; yT = v_ys*T
p_com_T = np.stack((xT, yT, w_zs*T), axis=1)        # (BATCH,3)

# Compute foot0, foot1 in world frame (BATCH×4×3)
# rotate default_foot_pos by heading at t=0 and t=T
def rotate_batch(pts, thetas):
    # pts: (4,3) or (B,4,3), thetas: (B,)
    # returns (B,4,3)
    B = thetas.shape[0]
    R = np.stack([
      [ np.cos(thetas), -np.sin(thetas), np.zeros(B)],
      [ np.sin(thetas),  np.cos(thetas), np.zeros(B)],
      [ np.zeros(B),     np.zeros(B),    np.ones(B)]
    ], axis=-1).reshape(B,3,3)
    # broadcast pts to (B,4,3)
    P = np.broadcast_to(default_foot_pos[None], (B,4,3))
    return (R[:,None] @ P[...,None]).squeeze(-1)

# world-frame foot positions at t=0 and t=T
foot0 = p_com_0[:,None,:] + rotate_batch(default_foot_pos, p_com_0[:,2])
foot1 = p_com_T[:,None,:] + rotate_batch(default_foot_pos, p_com_T[:,2])

# ── 6) TIME-STEPPING LOOP ─────────────────────────────────────────────────
for i, t in enumerate(ts):
    # gait phase and lift z
    phase = np.clip((t/T - 0.25)*2, 0, 1)
    up    = (phase <= 0.5)
    # bezier for z
    zb = np.where( phase<=0.5,
        (2*phase)**3 + 3*(2*phase)**2*(1-2*phase),
        ((2*phase-1)**3 + 3*(2*phase-1)**2*(2- (2*phase-1)) )
    )
    z     = np.where(up, zb * swing_h, zb * 0)  # (BATCH,)
    # p_com at time t
    if abs(w_zs).any():
        xt = (v_xs*np.sin(w_zs*t)+v_ys*(np.cos(w_zs*t)-1))/w_zs
        yt = (v_xs*(1-np.cos(w_zs*t))+v_ys*np.sin(w_zs*t))/w_zs
    else:
        xt = v_xs*t; yt = v_ys*t
    thetas = w_zs*t                         # (BATCH,)
    p_com   = np.stack((xt, yt, thetas), axis=1)  # (BATCH,3)

    # interpolate foot global pos
    fpw = foot0*(1-phase) + foot1*(phase)  # (B,4,3)
    # body-frame: Rᵀ (fpw - p_com)
    Rts = rotate_batch(fpw - p_com[:,None,:], -thetas)  # (B,4,3)
    Rts[...,2] += z                             # add lift

    # store foot refs
    foot_refs[:, i, :, :] = Rts

    # run IK on GPU
    inp = torch.from_numpy(Rts.reshape(BATCH,12).T)\
                .to(device='cuda', dtype=torch.double)
    torch.cuda.synchronize()
    fn_gpu.evaluate([inp])
    torch.cuda.synchronize()
    Q = fn_gpu.outputs_sparse[0].cpu().numpy()  # (BATCH,12)

    q_refs[:, i, :] = Q

    if i%10==0:
        print(f"Step {i+1}/{N}")

# ── 7) SAVE RESULTS ───────────────────────────────────────────────────────
os.makedirs("references", exist_ok=True)
np.save("go_2/references/go2_vxs.npy",        v_xs)
np.save("go_2/references/go2_vys.npy",        v_ys)
np.save("go_2/references/go2_wzs.npy",        w_zs)
np.save("go_2/references/go2_reference_ts.npy", ts)
np.save("go_2/references/go2_reference_qs.npy",  q_refs)
np.save("go_2/references/go2_reference_foot_refs.npy", foot_refs)

print("Saved batch references for", BATCH, "robots.")
