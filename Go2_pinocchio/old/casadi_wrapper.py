# gen_reference_casadi_debug.py

import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin
import numpy as np

# ----------------------------
# 1. Numeric Pinocchio Setup
# ----------------------------

URDF_PATH = "Go2_pinocchio/go2_original.urdf"

# Build the numeric Pinocchio model/data to extract q_init and default foot poses
model_pin = pin.buildModelFromUrdf(URDF_PATH)
data_pin = model_pin.createData()

# “Neutral” posture (in degrees) exactly from go2_reference_pin.py
q_init_numpy = np.deg2rad(np.array([
    -5.7,  45.8, -86.0,
    +5.7,  45.8, -86.0,
    -5.7,  57.3, -86.0,
    +5.7,  57.3, -86.0
]))
pin.forwardKinematics(model_pin, data_pin, q_init_numpy)
pin.updateFramePlacements(model_pin, data_pin)

# Foot frames
frame_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
frame_ids   = [model_pin.getFrameId(name) for name in frame_names]

# Numeric default foot positions (4×3)
default_foot_pos_num = np.vstack([
    data_pin.oMf[fid].translation.copy() for fid in frame_ids
])

# ----------------------------
# 2. CasADi Symbolic Setup
# ----------------------------

# Cast constants to SX
q_init    = ca.SX(q_init_numpy)                 # (12×1)
p_foot0   = ca.SX(default_foot_pos_num)         # (4×3)

# Build CasADi-enabled Pinocchio model/data
cmodel = cpin.Model(model_pin)
cdata  = cmodel.createData()

nq = model_pin.nq  # should be 12

# Discretization
N = 100  # number of time steps

# IK parameters from original Python code
damping   = 1e-4
ik_iters  = 100
small_wz  = 1e-2    # threshold for “near zero” w_z

# ----------------------------
# 3. Define CasADi Inputs
# ----------------------------

v_x     = ca.SX.sym("v_x")      # forward velocity
v_y     = ca.SX.sym("v_y")      # lateral velocity
w_z     = ca.SX.sym("w_z")      # yaw rate
swing_h = ca.SX.sym("swing_h")  # swing height
T       = ca.SX.sym("T")        # total duration

# ----------------------------
# 4. Precompute ts (time stamps)
# ----------------------------

ts = ca.SX(N, 1)
for i in range(N):
    ts[i] = (T * i) / (N - 1)

# ----------------------------
# 5. Precompute p_foot1 Symbolically
# ----------------------------

theta_end = w_z * T

x_end = ca.if_else(
    ca.fabs(w_z) < small_wz,
    v_x * T,
    (v_x * ca.sin(theta_end) + v_y * (ca.cos(theta_end) - 1)) / w_z
)
y_end = ca.if_else(
    ca.fabs(w_z) < small_wz,
    v_y * T,
    (v_x * (1 - ca.cos(theta_end)) + v_y * ca.sin(theta_end)) / w_z
)

c_end = ca.cos(theta_end)
s_end = ca.sin(theta_end)
R_end = ca.SX(3, 3)
R_end[0, 0] =  c_end;  R_end[0, 1] = -s_end;  R_end[0, 2] = 0
R_end[1, 0] =  s_end;  R_end[1, 1] =  c_end;  R_end[1, 2] = 0
R_end[2, 0] =     0  ;  R_end[2, 1] =     0  ;  R_end[2, 2] = 1

# Rotate default foot positions (4×3)
p_rotated = (R_end @ p_foot0.T).T  # (4×3)

# Build matrix of [x_end, y_end, 0] repeated 4×
p_end_row   = ca.vertcat(x_end, y_end, 0).T  # (1×3)
p_end_mat   = ca.repmat(p_end_row, 4, 1)     # (4×3)

p_foot1 = p_end_mat + p_rotated  # (4×3)

# ----------------------------
# 6. Allocate Outputs
# ----------------------------

q_ref         = ca.SX.zeros(nq, N)    # (12×100)
foot_ref_flat = ca.SX.zeros(12, N)    # flatten(4×3) → (12×1) per column

# ----------------------------
# 7. Main Loop Over Time Steps (with debug prints)
# ----------------------------

q_prev = q_init  # initial guess

print("=== Starting symbolic graph construction (this may take a while) ===")
for i in range(N):
    print(f"[DEBUG] Time step i = {i+1} / {N}")

    # Current CoM pose
    t_i     = ts[i]
    theta_i = w_z * t_i

    x_i = ca.if_else(
        ca.fabs(w_z) < small_wz,
        v_x * t_i,
        (v_x * ca.sin(theta_i) + v_y * (ca.cos(theta_i) - 1)) / w_z
    )
    y_i = ca.if_else(
        ca.fabs(w_z) < small_wz,
        v_y * t_i,
        (v_x * (1 - ca.cos(theta_i)) + v_y * ca.sin(theta_i)) / w_z
    )

    # Phase & clamp
    phase_raw     = (t_i / T - 0.25) * 2
    phase_clamped = ca.fmin(ca.fmax(phase_raw, 0), 1)

    # Cubic Bézier for vertical z
    u1   = 2 * phase_clamped
    bez1 = u1**3 + 3 * (u1**2 * (1 - u1))
    z1   = bez1 * swing_h

    u2   = 2 * phase_clamped - 1
    bez2 = u2**3 + 3 * (u2**2 * (1 - u2))
    z2   = swing_h - (bez2 * swing_h)
    z    = ca.if_else(phase_clamped <= 0.5, z1, z2)

    # Horizontal Kinematics Bézier
    bez_sp      = phase_clamped**3 + 3 * (phase_clamped**2 * (1 - phase_clamped))
    foot_global = p_foot0 + (p_foot1 - p_foot0) * bez_sp  # (4×3)

    # Rotate into body frame
    c_i    = ca.cos(theta_i);  s_i = ca.sin(theta_i)
    R_body = ca.SX(3, 3)
    R_body[0, 0] =  c_i;  R_body[0, 1] =  s_i;  R_body[0, 2] = 0
    R_body[1, 0] = -s_i;  R_body[1, 1] =  c_i;  R_body[1, 2] = 0
    R_body[2, 0] =   0 ;  R_body[2, 1] =   0 ;  R_body[2, 2] = 1

    com_row     = ca.vertcat(x_i, y_i, 0).T   # (1×3)
    com_mat     = ca.repmat(com_row, 4, 1)    # (4×3)
    diff_global = foot_global - com_mat       # (4×3)
    body_pos    = (R_body @ diff_global.T).T  # (4×3)
    # Add swing height z to Z-column
    body_pos = ca.horzcat(
        body_pos[:, 0:1],
        body_pos[:, 1:2],
        body_pos[:, 2:3] + z
    )  # still (4×3)

    # --- Inverse Kinematics for each foot (with debug prints) ---
    q = q_prev  # initial guess for this time step
    for k in range(4):
        print(f"    [DEBUG]   Foot k = {k+1} / 4 at time step {i+1}")
        fk = ca.vertcat(body_pos[k, 0], body_pos[k, 1], body_pos[k, 2])  # (3×1)

        for it in range(ik_iters):
            # Print only every 20 iterations
            if it % 20 == 0:
                print(f"        [DEBUG]     IK iter = {it+1} / {ik_iters} for foot {k+1}")

            # 1) Forward kinematics
            cpin.framesForwardKinematics(cmodel, cdata, q)

            # 2) Current foot pos
            p_cur = cdata.oMf[frame_ids[k]].translation

            # 3) Position error
            err = fk - p_cur

            # 4) Jacobian (6×nq) → take top 3 rows
            J6    = cpin.computeFrameJacobian(
                        cmodel, cdata, q,
                        frame_ids[k],
                        cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                    )
            J_pos = J6[0:3, :]  # (3×12)

            # 5) Damped least-squares update
            A    = J_pos @ J_pos.T + damping * ca.SX_eye(3)  # (3×3)
            invA = ca.solve(A, ca.SX_eye(3))                # (3×3)
            J_dls = J_pos.T @ invA                           # (12×3)

            # 6) Update q
            q = q + J_dls @ err

        # End of IK iterations for foot k

    # End of 4-foot loop → q holds joint angles (12×1) for this time step

    # Store q into q_ref[:, i]
    for j in range(nq):
        q_ref[j, i] = q[j]

    # Flatten body_pos (4×3) → (12×1) and store in foot_ref_flat[:, i]
    flat = ca.reshape(body_pos.T, 12, 1)
    for j in range(12):
        foot_ref_flat[j, i] = flat[j]

    q_prev = q  # update initial guess for next time step

# End of time loop
print("=== Finished building symbolic graph ===")

# ----------------------------
# 8. Build & Save CasADi Function
# ----------------------------
generate_reference = ca.Function(
    "generate_reference",
    [v_x, v_y, w_z, swing_h, T],
    [ts, q_ref, foot_ref_flat]
)

print("=== Saving .casadi file to generate_reference.casadi ===")
generate_reference.save("generate_reference.casadi")
print("=== Done. ===")
