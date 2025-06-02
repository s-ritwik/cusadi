# IK_go2_casadi_gen.py
import casadi as ca

# 0) Robot geometry from your XML / URDF (all in meters)
#    hip_offset is lateral distance from body center to each hip
hip_offset = 0.0465  
L_thigh = 0.213    # thigh
L_shin  = 0.213    # shin

# 1) Symbolic variables
#    q = [q0…q11] is all 12 leg joints:
#      [FL_abd, FL_hip, FL_knee,
#       FR_abd, FR_hip, FR_knee,
#       RL_abd, RL_hip, RL_knee,
#       RR_abd, RR_hip, RR_knee]
q = ca.SX.sym('q', 12, 1)

#    p_target is 4*(x,y,z) stacked
p_target = ca.SX.sym('p', 12, 1)

# 2) Helper: single-leg FK in CasADi
def leg_fk(x_off, y_off, q_abd, q_hip, q_knee):
    """
    Returns foot pos [x,y,z] for one leg.
    x_off, y_off = hip location in body frame.
    q_abd: ab/adduction about body‐Y
    q_hip: hip pitch about body‐X
    q_knee: knee pitch about leg frame Y
    """
    # Rotate hip ab/adduction (about Y) at the hip mount:
    R_abd = ca.vertcat(
      ca.horzcat(ca.cos(q_abd), 0, -ca.sin(q_abd)),
      ca.horzcat(0,            1, 0),
      ca.horzcat(ca.sin(q_abd), 0,  ca.cos(q_abd))
    )
    # rotate hip pitch (about X)
    R_hip = ca.vertcat(
      ca.horzcat(1, 0,             0),
      ca.horzcat(0, ca.cos(q_hip), -ca.sin(q_hip)),
      ca.horzcat(0, ca.sin(q_hip),  ca.cos(q_hip))
    )
    # knee pitch about Y in thigh frame:
    R_knee = ca.vertcat(
      ca.horzcat(ca.cos(q_knee), 0, -ca.sin(q_knee)),
      ca.horzcat(0,             1, 0),
      ca.horzcat(ca.sin(q_knee), 0,  ca.cos(q_knee))
    )

    # hip to thigh translation along body X by 0 (no offset in X), then along Z:
    p_hip = ca.vertcat(x_off, y_off, 0)
    # thigh end relative to hip: along body-frame X by L_thigh in hip-rotated frame
    p_thigh = R_abd @ R_hip @ ca.vertcat(L_thigh, 0, 0)
    # shin end relative to thigh end:
    p_shin = R_abd @ R_hip @ R_knee @ ca.vertcat(L_shin, 0, 0)

    # foot position:
    return p_hip + p_thigh + p_shin  # 3×1

# 3) Build the 4 feet → 12×1 vector
feet = []
# offsets: FL (+x,+y), FR(+x, -y), RL(-x,+y), RR(-x,-y)
offs = [( 0.1934,  hip_offset),   # approx from your XML
        ( 0.1934, -hip_offset),
        (-0.1934,  hip_offset),
        (-0.1934, -hip_offset)]
for i,(xo,yo) in enumerate(offs):
    q0 = q[3*i + 0]  # ab/adduction
    q1 = q[3*i + 1]  # hip pitch
    q2 = q[3*i + 2]  # knee pitch
    feet.append(leg_fk(xo, yo, q0, q1, q2))
feet_pos = ca.vertcat(*feet)         # 12×1

# 4) residual = feet_pos - p_target
resid = feet_pos - p_target          # 12×1

# 5) residual+Jacobian
F = ca.Function('F', [q, p_target],
                [resid, ca.jacobian(resid, q)],
                ['q','p'], ['r','J'])

# 6) Newton rootfinder
ik = ca.rootfinder('fn_IK_go2', 'newton', F)

# 7) wrap and save
q_sol = ik(p_target)
fn = ca.Function('fn_IK_go2', [p_target], [q_sol],
                 ['foot_pos'], ['q'])
fn.save('fn_IK_go2.casadi')
print("Saved fn_IK_go2.casadi")
