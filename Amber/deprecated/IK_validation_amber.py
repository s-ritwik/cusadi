import numpy as np
import casadi as ca
import pinocchio as pin

def test_amber_ik_fk():
    # 1) Load IK helper
    ik_fun = ca.Function.load("amber_reference_step.casadi")
    
    # 2) Build the Pinocchio model & data for FK
    URDF = "Amber/amber_free.urdf"              # adjust path if needed
    model = pin.buildModelFromUrdf(URDF)
    data  = model.createData()
    
    # frame names must match your URDF
    FOOT_FRAMES = ["left_toe", "right_toe"]
    frame_ids   = [model.getFrameId(f) for f in FOOT_FRAMES]
    
    # 3) Define a little test
    phase    = 0.15                          # some phase in [0,1)
    foot_x   = np.array([ 0.8, -0.2 ])     # desired x positions of left & right toes
    x_com    = 0.0
    z_com    = 0.33                          # nominal stand height
    z_swing  = 0.10                          # swing‐height
    q_guess  = np.zeros(4)                   # initial angle guess
    
    # 4) Call IK
    #    returns q_ref (4,) and foot_body_flat which we ignore here
    q_ref_casadi, _ = ik_fun(
        phase,
        foot_x,
        x_com,
        z_com,
        z_swing,
        q_guess
    )
    q_ref = np.array(q_ref_casadi).flatten()
    print("angle by IK:",q_ref)
    # 5) Inject a tiny error
    q_noisy = q_ref + np.random.normal(scale=1e-2, size=4)
    print("noisy angle for FK:",q_noisy)
    # 6) Numeric FK → get world‐frame toe transforms
    pin.forwardKinematics(model, data, q_noisy)
    pin.updateFramePlacements(model, data)
    fk_pos = []
    for fid in frame_ids:
        xyz = data.oMf[fid].translation
        fk_pos.append([xyz[0], xyz[2]])    # pick x,z coords only
    fk_pos = np.array(fk_pos)  # shape (2,2)
    
    # 7) Print a comparison
    print("  ref foot-x’s   :", foot_x)
    print("  FK-recovered x’s:", fk_pos[:,0])
    print("  FK-recovered z’s:", fk_pos[:,1])
    print("\nIf everything’s wired up correctly, FK-recovered x’s should be ≈ the refs.")
    
if __name__=="__main__":
    test_amber_ik_fk()
