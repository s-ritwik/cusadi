# visualize_refs_batch.py
import os, copy, time
import xml.etree.ElementTree as ET
import mujoco, mujoco.viewer
import numpy as np

# ——————————————————————————————————————————————————————————————————
# Paths to your data
ORIG_XML   = "go_2/go2_fixed.xml"
MULTI_XML  = "go_2/multi_go2_fixed.xml"
REF_Q_PATH = "go_2/references/go2_reference_qs.npy"
REF_T_PATH = "go_2/references/go2_reference_ts.npy"
SPACING    = 1.5        # metres between robot bases
# ——————————————————————————————————————————————————————————————————

# 1) Load the grid of references and flatten to (BATCH, N, 12)
q_grid = np.load(REF_Q_PATH)   # shape (nx, ny, nw, N, 12)
ts     = np.load(REF_T_PATH)   # shape (N,)
nx, ny, nw, N, _ = q_grid.shape
q_refs = q_grid.reshape(-1, N, 12)
BATCH   = q_refs.shape[0]
print(N)
print(f"Loaded references for {nx}×{ny}×{nw}={BATCH} trajectories")

# 2) Build merged XML if needed
if not os.path.exists(MULTI_XML):
    tree = ET.parse(ORIG_XML)
    root = tree.getroot()
    # strip out keyframe, sensors, actuators
    for tag in ("keyframe","sensor","actuator"):
        for node in root.findall(tag):
            root.remove(node)
    world = root.find("worldbody")
    torso = world.find("body")
    world.remove(torso)
    base_z = float(torso.get("pos").split()[2])

    for i in range(BATCH):
        robot = copy.deepcopy(torso)
        # rename only the top-level body to guarantee uniqueness
        robot.set("name", f"torso_{i}")
        # remove all other 'name' attributes in its subtree
        for elem in robot.iter():
            if elem is robot: continue
            if "name" in elem.attrib:
                del elem.attrib["name"]
        # spread them out in X
        robot.set("pos", f"{i*SPACING} 0 {base_z}")
        world.append(robot)

    tree.write(MULTI_XML)
    print("Wrote merged XML:", MULTI_XML)

# 3) Load the combined model
model = mujoco.MjModel.from_xml_path(MULTI_XML)
data  = mujoco.MjData(model)

# 4) Animate all BATCH robots in lock-step
dt      = 1/60
t       = 0.0
N_steps = ts.shape[0]
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # pick frame index
        idx = int((t / ts[-1]) * N_steps) % N_steps
        # apply each robot's joint angles
        for i in range(BATCH):
            start = 12*i
            data.qpos[start:start+12] = q_refs[i, idx]
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(dt)
        t += dt
