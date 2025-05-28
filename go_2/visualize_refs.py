import mujoco.viewer
import mujoco
import numpy as np
import time


v_x = 1.2
v_y = .5
w_z = 0.25


def yaw2quat(yaw):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    return np.array([cy, 0., 0., sy])  # (w, x, y, z)


if __name__ == "__main__":
    v_xs = np.load("references/go2_vxs.npy")
    v_ys = np.load("references/go2_vys.npy")
    w_zs = np.load("references/go2_wzs.npy")
    ts = np.load("references/go2_reference_ts.npy")
    q_refs = np.load("references/go2_reference_qs.npy")
    foot_refs = np.load("references/go2_reference_foot_refs.npy")

    # Load the XML model
    model = mujoco.MjModel.from_xml_path("go2.xml")
    data = mujoco.MjData(model)

    # 1. Load keyframe and set the robot state
    keyframe_id = 0  # Assuming the keyframe ID you want to use is 0
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    mujoco.mj_forward(model, data)  # Perform forward kinematics
    print("Loaded keyframe:", keyframe_id)

    foot_names = ["FL_foot_site", "FR_foot_site", "RL_foot_site", "RR_foot_site"]
    base_foot_positions = {}
    foot_ids = []
    for foot in foot_names:
        foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, foot)
        foot_ids.append(foot_id)
        site_pos = data.site_xpos[foot_id]
        base_foot_positions[foot_id] = site_pos.copy()
        print(f"Position of {foot}: {site_pos}")

    # Find interpolation indices
    vx_low = np.where(v_xs <= v_x)[0][-1]   # Last index less than v_x
    vx_high = np.argmax(v_xs >= v_x)        # First index greater than v_x
    vy_low = np.where(v_ys <= v_y)[0][-1]  # Last index less than v_x
    vy_high = np.argmax(v_ys >= v_y)  # First index greater than v_x
    wz_low = np.where(w_zs <= w_z)[0][-1]  # Last index less than v_x
    wz_high = np.argmax(w_zs >= w_z)  # First index greater than v_x

    z = 0.33
    T = max(ts)
    phase_offset = np.array([0, T / 2, T / 2, 0])
    t = 0
    dt = 1 / 60
    sim_rate = 0.2
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # TODO: interpolate q out of data
            t0 = time.time()
            t += sim_rate * dt
            theta = w_z * t
            x = (v_x * np.sin(w_z * t) + v_y * (np.cos(w_z * t) - 1)) / w_z if abs(w_z) > 1e-3 else v_x * t
            y = (v_x * (1 - np.cos(w_z * t)) + v_y * np.sin(w_z * t)) / w_z if abs(w_z) > 1e-3 else v_y * t

            q = np.zeros((12,))

            vx_inds = np.array([vx_low] * 8 + [vx_high] * 8)
            vy_inds = np.tile(np.array([vy_low] * 4 + [vy_high] * 4), 2)
            wz_inds = np.tile(np.array([wz_low] * 2 + [wz_high] * 2), 4)
            for foot_id in range(4):
                ts_low = np.where(ts <= (t + phase_offset[foot_id]) % T)[0][-1]  # Last index less than v_x
                ts_high = np.argmax(ts >= (t + phase_offset[foot_id]) % T)  # First index greater than v_x

                t_inds = np.tile(np.array([ts_low, ts_high]), 8)

                vecs = np.stack([v_xs[vx_inds], v_ys[vy_inds], w_zs[wz_inds], ts[t_inds]])
                des_vec = np.array([v_x, v_y, w_z, t])
                dist = np.linalg.norm(vecs - des_vec[:, None], axis=0) + 1e-4
                weights = 1 / dist
                weights = weights / np.sum(weights, keepdims=True)

                q[3*foot_id: 3*(foot_id+1)] = np.sum(weights[:, None] * q_refs[vx_inds, vy_inds, wz_inds, t_inds, 3*foot_id:3*(foot_id+1)], axis=0)

            quat = yaw2quat(theta)
            data.qpos[:] = np.hstack((np.array((x, y, z)), quat, q))
            mujoco.mj_forward(model, data)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            time.sleep(max(0, dt - (time.time() - t0)))
