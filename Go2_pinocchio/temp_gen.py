# go2_IK_casadi_wrapper.py

import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin  # NEW import

# -------------------------------
# 1) Build the Pinocchio CasADi model
# -------------------------------
GO2_URDF = "Go2_pinocchio/go2_original.urdf"

# Build the original (numeric) Pinocchio model
model_num = pin.buildModelFromUrdf(GO2_URDF)
# Build CasADi model
cmodel = cpin.Model(model_num)   # Correct
cdata  = cmodel.createData()

# Foot frame names (should match your URDF)
FOOT_FRAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
# Get their numeric frame IDs (used by Pinocchio)
frame_ids = [model_num.getFrameId(n) for n in FOOT_FRAMES]

# IK constants
DAMPING = ca.SX(1e-4)   # damping for DLS inverse
IK_ITERS = 20          # max iterations per foot
IK_TOL = ca.SX(1e-6)    # tolerance (unused here since we run fixed # of iterations)

# -------------------------------
# 2) Define the CasADi SX inputs
# -------------------------------
t_i           = ca.SX.sym("t_i")                       # current time (scalar)
phase         = ca.SX.sym("phase")                     # swing phase in [0,1]
z_i           = ca.SX.sym("z_i")                       # swing height at this time
foot_pos_world = ca.SX.sym("foot_pos_world", 4, 3)     # 4×3: world‐frame foot positions
x             = ca.SX.sym("x")                         # CoM x (scalar)
y             = ca.SX.sym("y")                         # CoM y (scalar)
theta         = ca.SX.sym("theta")                     # CoM yaw (scalar)

# -------------------------------
# 3) Build body‐frame rotation from yaw (world→body)
# -------------------------------
c = ca.cos(theta)
s = ca.sin(theta)
R_body = ca.vertcat(
    ca.horzcat( c,  s, ca.SX(0)),
    ca.horzcat(-s,  c, ca.SX(0)),
    ca.horzcat( ca.SX(0), ca.SX(0), ca.SX(1))
)

# -------------------------------
# 4) Translate & rotate foot pos → body frame, then add swing height
# -------------------------------
# 4.1) Build CoM‐pos vector (x,y,0) and subtract from each world foot pos
com_pos      = ca.vertcat(x, y, 0)            # 3×1
foot_rel     = foot_pos_world - ca.repmat(com_pos.T, 4, 1)  # 4×3

# 4.2) Rotate each relative foot vector into body frame
foot_body_no_z = (R_body @ foot_rel.T).T      # 4×3: still missing z‐offset

# 4.3) Add the swing height z_i to the z‐component of every foot
#       (only the third column entry)
z_column      = ca.repmat(ca.vertcat(0, 0, z_i).T, 4, 1)  # 4×3
foot_ref_i    = foot_body_no_z + z_column                  # 4×3

# -------------------------------
# 5) Initialize q_current from a neutral pose (CasADi SX)
# -------------------------------
#    The same "neutral" joint angles used in your original .py
q_init_deg = [
    -5.7, 45.8, -86.0,
     +5.7, 45.8, -86.0,
    -5.7, 57.3, -86.0,
     +5.7, 57.3, -86.0
]
# Convert to radians and pack into an SX vector
q_init_rad = [deg * ca.SX(3.141592653589793 / 180.0) for deg in q_init_deg]
q_current = ca.vertcat(*[ca.SX(val) for val in q_init_rad])  # 12×1 SX

# -------------------------------
# 6) Perform multi‐foot IK with Pinocchio (all SX)
# -------------------------------
# We'll solve each foot in sequence, exactly like `inverse_kinematics_pinocchio`,
# but using cmodel/cdata and SX everywhere.

for leg_idx in range(4):
    # 6.1) Identify the three joint indices for this leg in q_current
    #     (FL→[0,1,2], FR→[3,4,5], RL→[6,7,8], RR→[9,10,11])
    if leg_idx == 0:
        idx = slice(0, 3)
    elif leg_idx == 1:
        idx = slice(3, 6)
    elif leg_idx == 2:
        idx = slice(6, 9)
    else:
        idx = slice(9, 12)

    # 6.2) Fixed‐count Damped Least Squares iterations for this foot
    for _ in range(IK_ITERS):
        # 6.2.1) Compute forward kinematics up to all joints (SX)
        # pin.forwardKinematics(cmodel, cdata, q_current)
        # pin.updateFramePlacements(cmodel, cdata)
        cpin.forwardKinematics(cmodel, cdata, q_current)
        cpin.updateFramePlacements(cmodel, cdata)
        # 6.2.2) Current foot position (SX) from cdata
        p_cur = cdata.oMf[frame_ids[leg_idx]].translation  # 3×1 SX

        # 6.2.3) Desired foot position (SX) is foot_ref_i[leg_idx,:].T
        p_des = foot_ref_i[leg_idx, :].T  # 3×1 SX

        # 6.2.4) Position error
        err = p_des - p_cur               # 3×1 SX

        # 6.2.5) Build the 6×12 Jacobian for that foot, then take top‐3 rows
        J6    = cpin.computeFrameJacobian(
                    cmodel, cdata, q_current, frame_ids[leg_idx],
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )                              # 6×12 SX
        J_pos = J6[0:3, :]                            # 3×12 SX

        # 6.2.6) Shorten J_pos to the 3×3 block for this leg's joints only
        #         (J_leg maps leg‐joint increments → foot position)
        #         That is: pick columns idx.start..idx.stop from J_pos
        J_leg = J_pos[:, idx]                         # 3×3 SX

        # 6.2.7) Damped least‐squares: Δq_leg = J_leg^T * (J_leg J_leg^T + λ I)^(-1) * err
        JJT   = J_leg @ J_leg.T                       # 3×3 SX
        inv_term = ca.inv(JJT + DAMPING * ca.SX_eye(3))  # 3×3 SX
        dq_leg   = J_leg.T @ (inv_term @ err)         # 3×1 SX

        # 6.2.8) Update only these three joint entries in q_current
        q_vec = []
        for qi in range(12):
            if idx.start <= qi < idx.stop:
                local_idx = qi - idx.start
                q_vec.append(q_current[qi] + dq_leg[local_idx])
            else:
                q_vec.append(q_current[qi])
        q_current = ca.vertcat(*q_vec)  # new 12×1 SX

        # (We could break early if norm(err)<IK_TOL, but we unroll all IK_ITERS)

# After all four feet have been updated, q_current is our IK solution
q_ref_i = q_current   # 12×1 SX

# -------------------------------
# 7) Build and save the CasADi Function
# -------------------------------
go2_single_step = ca.Function(
    "go2_single_step",
    [t_i, phase, z_i, foot_pos_world, x, y, theta],
    [t_i, q_ref_i, foot_ref_i]
)
# Save it to disk as “.casadi”
go2_single_step.save("go2_IK_step.casadi")
