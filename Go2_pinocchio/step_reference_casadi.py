# generate_step_casadi.py

import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin
# -----------------------------------
# 1. Build the (numeric) Pinocchio model
# -----------------------------------

# Path to your URDF
URDF_PATH = "Go2_pinocchio/go2_original.urdf"

# Build standard Pinocchio model to extract joint dims, frame IDs, etc.
model_pin = pin.buildModelFromUrdf(URDF_PATH)
data_pin = model_pin.createData()

# Foot frame names must match those in your URDF
frame_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
frame_ids = [model_pin.getFrameId(name) for name in frame_names]  # e.g. [id0, id1, id2, id3]

# Number of joints (should be 12 for Go2)
nq = model_pin.nq  # 12

# -----------------------------------
# 2. Build the CasADi‐compatible Pinocchio model
# -----------------------------------

# This clones the numeric model into a CasADi SX model
cmodel = cpin.Model(model_pin)
cdata  = cmodel.createData()

# -----------------------------------
# 3. Define all CasADi SX symbols for one time‐step
# -----------------------------------

# Scalar inputs for this single “step” function:
phase   = ca.SX.sym("phase")       # scalar in [0,1], clipped already
x_i     = ca.SX.sym("x_i")         # CoM x at this timestep
y_i     = ca.SX.sym("y_i")         # CoM y at this timestep
theta   = ca.SX.sym("theta")       # CoM heading at this timestep

# Matrices for the “start‐foot” and “end‐foot” global positions:
#   p_foot0, p_foot1 ∈ ℝ^{4×3}
p_foot0 = ca.SX.sym("p_foot0", 4, 3)
p_foot1 = ca.SX.sym("p_foot1", 4, 3)

# Swing‐height (scalar), to lift foot in z
swing_h = ca.SX.sym("swing_h")

# Initial‐guess for IK at this step (12×1)
q_init  = ca.SX.sym("q_init", nq)

# -----------------------------------
# 4. Inside‐loop logic (becomes the graph)
# -----------------------------------

# 4.1. Compute foot‐height “z” via cubic Bézier:
#
#   if phase ≤ 0.5:
#     u1 = 2*phase
#     bez1 = u1^3 + 3*(u1^2*(1-u1))
#     z1 = swing_h * bez1
#   else:
#     u2 = 2*phase - 1
#     bez2 = u2^3 + 3*(u2^2*(1-u2))
#     z2 = swing_h – swing_h * bez2
#   z = (phase ≤ 0.5 ? z1 : z2)
u1    = 2 * phase
bez1  = u1**3 + 3 * (u1**2 * (1 - u1))
z1    = swing_h * bez1

u2    = 2 * phase - 1
bez2  = u2**3 + 3 * (u2**2 * (1 - u2))
z2    = swing_h - swing_h * bez2

z = ca.if_else(phase <= 0.5, z1, z2)  # scalar

# 4.2. Compute “horizontal” foot‐trajectory in **global frame** (4×3):
#   foot_global = p_foot0 + (p_foot1 - p_foot0) * bez_sp
bez_sp      = phase**3 + 3 * (phase**2 * (1 - phase))  # scalar
foot_global = p_foot0 + (p_foot1 - p_foot0) * bez_sp   # (4×3)

# 4.3. Transform foot_global → foot_body (4×3):
#    R_body = [ [ cos θ,  sin θ, 0 ];
#               [ -sin θ, cos θ, 0 ];
#               [   0,       0,   1 ] ]
c = ca.cos(theta)
s = ca.sin(theta)
R_body = ca.SX(3, 3)
R_body[0, 0] =  c;  R_body[0, 1] =  s;  R_body[0, 2] = 0
R_body[1, 0] = -s;  R_body[1, 1] =  c;  R_body[1, 2] = 0
R_body[2, 0] =  0;  R_body[2, 1] =  0;  R_body[2, 2] = 1

# Build a (4×3) matrix “com_mat” where each row is [x_i, y_i, 0]
com_row = ca.vertcat(x_i, y_i, 0).T          # (1×3)
com_mat = ca.repmat(com_row, 4, 1)           # (4×3)

# Subtract CoM XY → Δ_global = foot_global - com_mat
delta_global = foot_global - com_mat         # (4×3)

# Rotate into body frame: foot_body = (R_body @ delta_globalᵀ)ᵀ
foot_body = (R_body @ delta_global.T).T      # (4×3)

# Finally, add the swing‐height z to the 3rd column of foot_body
# (i.e. foot_body[k, 2] += z for each of the 4 feet)
fz         = foot_body[:, 2] + z             # (4×1)
foot_body  = ca.horzcat(foot_body[:, 0],      # (4×3) after reassembly
                        foot_body[:, 1],
                        fz)

# 4.4. Inverse kinematics (Damped Least‐Squares) for each of 4 feet in SEQUENCE
q = q_init                                  # start from the provided initial guess

damping  = 1e-4
ik_iters = 100
tol = 1e-6
convergence_flags = []  # initialize empty list to store flags per foot

for k in range(4):
    # target foot‐position k:
    fk = ca.vertcat(foot_body[k, 0],
                    foot_body[k, 1],
                    foot_body[k, 2])         # (3×1) SX

    for _ in range(ik_iters):
        # 1) symbolically forward‐kinematics:
        # cpin.framesForwardKinematics(cmodel, cdata, q)
        cpin.forwardKinematics(cmodel, cdata, q)
        cpin.computeJointJacobians(cmodel, cdata, q)
        cpin.framesForwardKinematics(cmodel, cdata, q)
        cpin.updateFramePlacements(cmodel, cdata)
        p_cur = cdata.oMf[frame_ids[k]].translation
        # 2) current foot‐position from Pinocchio:
        p_cur = cdata.oMf[frame_ids[k]].translation  # (3×1) SX

        # 3) position error:
        err = fk - p_cur                              # (3×1)
        norm_e  = ca.norm_2(err)                  # symbolic norm

        # 4) 6×12 Jacobian, then take position rows:
        J6    = cpin.computeFrameJacobian(
                    cmodel, cdata, q,
                    frame_ids[k],
                    cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )                                    # (6×12) SX
        J_pos = J6[0:3, :]                             # (3×12) SX

        # 5) damped‐least‐squares step:
        A    = J_pos @ J_pos.T + damping * ca.SX_eye(3)  # (3×3)
        invA = ca.solve(A, ca.SX_eye(3))                 # (3×3)
        Jdls = J_pos.T @ invA                            # (12×3)
        delta = Jdls @ err                        # candidate change in q
        foot_converged = norm_e < tol  # symbolic boolean (SX scalar)

        # 6) update q ← q + Jdls · err
        # q    = q + Jdls @ err                            # (12×1)
        q = ca.if_else(norm_e < tol, q, q + delta)
        convergence_flags.append(foot_converged)

    # end inner IK‐loop for foot k

# end for k in range(4)

# q is now the final joint solution (12×1)
q_out      = q

# 4.5. Flatten foot_body (4×3) → (12×1) in row‐major order
foot_flat  = ca.reshape(foot_body, 12, 1)  # (12×1)

# -----------------------------------
# 5. Pack everything into a CasADi function
# -----------------------------------

generate_step = ca.Function(
    "generate_step",
    [
        phase,    # scalar
        x_i,      # scalar
        y_i,      # scalar
        theta,    # scalar
        p_foot0,  # (4×3)
        p_foot1,  # (4×3)
        swing_h,  # scalar
        q_init    # (12×1)
    ],
    [
        q_out,    # (12×1)
        foot_flat, # (12×1)
        ca.vertcat(*convergence_flags)
    ]
)

# -----------------------------------
# 6. Save to .casadi
# -----------------------------------

generate_step.save("generate_step.casadi")

print("✅ generate_step.casadi has been created.")
