# IK_go2_gen.py
import casadi as ca

# —————————————————————————————————————————————
# 1. Robot geometry (in metres): thigh & shin lengths
L1 = 0.213   # thigh
L2 = 0.213   # shin
# —————————————————————————————————————————————

# 2. Symbolic foot‐position input: 4 legs × (x,y,z)
f = ca.SX.sym('f', 12, 1)  
# split into four 3×1 vectors
feet = [f[3*i:3*i+3] for i in range(4)]

qs = []
for p in feet:
    x, y, z = p[0], p[1], p[2]
    # project into sagittal plane
    r = ca.sqrt(x**2 + z**2)
    # ab/adduction
    q1 = ca.atan2(y, r)
    # hip flexion
    phi = ca.acos((L1**2 + r**2 - L2**2)/(2*L1*r))
    q2 = ca.atan2(z, x) - phi
    # knee
    q3 = ca.pi - ca.acos((L1**2 + L2**2 - r**2)/(2*L1*L2))
    qs += [q1, q2, q3]

# stack all 12 joint angles
q_out = ca.vertcat(*qs)

# 3. Build & save CasADi function
fn = ca.Function('fn_IK_go2',
                 [f],
                 [q_out],
                 ['foot_pos'], ['joint_ang'])
fn.save('fn_IK_go2.casadi')
print("Saved fn_IK_go2.casadi")
