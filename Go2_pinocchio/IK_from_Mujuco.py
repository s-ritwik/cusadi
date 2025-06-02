import mujoco
import numpy as np

# ---------- IK solver -------------------------------------------------
def mujoco_inverse_kinematics(target_xyz, foot_ids, model, data,
                              q_init=None, tol=1e-6, max_iter=500, damping=1e-4):
    if q_init is not None:
        data.qpos[:] = q_init
    mujoco.mj_forward(model, data)

    for idx, sid in enumerate(foot_ids):
        tgt = target_xyz[idx]
        for _ in range(max_iter):
            mujoco.mj_forward(model, data)
            cur   = data.site_xpos[sid].copy()
            err   = tgt - cur
            if np.linalg.norm(err) < tol:
                break
            Jpos  = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, Jpos, None, sid)
            Jpinv = Jpos.T @ np.linalg.inv(Jpos @ Jpos.T + damping*np.eye(3))
            data.qpos[:] += Jpinv @ err
        if np.linalg.norm(err) >= tol:
            print(f"[WARN] site {sid} not converged, ‖err‖={np.linalg.norm(err):.2e}")

    mujoco.mj_forward(model, data)
    return data.qpos.copy()

# ---------- IK → FK sanity check -------------------------------------
def ik_fk_sanity_check(target_xyz, foot_ids, model_path, q_start=None, **ik_opts):
    mdl  = mujoco.MjModel.from_xml_path(model_path)
    dat  = mujoco.MjData(mdl)

    q_sol = mujoco_inverse_kinematics(target_xyz, foot_ids, mdl, dat,
                                      q_init=q_start, **ik_opts)
    achieved = np.vstack([dat.site_xpos[sid].copy() for sid in foot_ids])
    errors   = np.linalg.norm(target_xyz - achieved, axis=1)

    # ---- print everything requested ----------------------------------
    print("\nSolved joint angles [rad] (legs FL → RR, hip-thigh-calf each):")
    print(q_sol.reshape(4, 3))

    print("\nForward-kinematics foot positions (x, y, z) [m]:")
    for name, pos in zip(["FL","FR","RL","RR"], achieved):
        print(f" {name}: {pos}")

    print("\nCartesian errors [m]:")
    for name, e in zip(["FL","FR","RL","RR"], errors):
        print(f" {name}: {e:.3e}")
    print(f"Max error  : {errors.max():.3e}")
    print(f"Mean error : {errors.mean():.3e}\n")

    return q_sol, achieved, errors

# ---------- example run ----------------------------------------------
if __name__ == "__main__":
    model_file = "go_2/go2_fixed.xml"
    site_names = ["FL_foot_site", "FR_foot_site", "RL_foot_site", "RR_foot_site"]

    mdl  = mujoco.MjModel.from_xml_path(model_file)
    sids = [mujoco.mj_name2id(mdl, mujoco.mjtObj.mjOBJ_SITE, n) for n in site_names]

    target_xyz = np.array([
        [ 0.180,  0.110, 0.126],
        [ 0.180, -0.110, 0.126],
        [-0.271,  0.111, 0.135],
        [-0.271, -0.111, 0.135],
    ])
    target_xyz = np.array([
    [ 0.17782152, 0.11044377, 0.12571125],   # FL
    [0.17782152, -0.11044377 ,0.12571125],   # FR
    [-0.27051568 , 0.11137226,  0.13496522],   # RL
    [-0.27051568  ,-0.11137226 , 0.13496522],   # RR
])  # shape (1,4,3) – one key-frame example

    ik_fk_sanity_check(target_xyz, sids, model_file,
                       tol=1e-6, max_iter=400, damping=1e-4)
