import casadi as ca
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
import os

# Constants - must match original
GO2_URDF = "Go2_pinocchio/go2_original.urdf"
FOOT_FRAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
DAMPING_PIN = 1e-4
IK_ITERS_PIN = 20
IK_TOL_PIN = 1e-6
OUTPUT_DIR = "src/casadi_functions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Build numerical model (for reference data)
model_pin = pin.buildModelFromUrdf(GO2_URDF)
data_pin = model_pin.createData()
frame_ids = [model_pin.getFrameId(name) for name in FOOT_FRAMES]

# Neutral configuration
q_init_np = pin.neutral(model_pin)
q_init_np[:] = np.deg2rad([
    -5.7, 45.8, -86.0,
    +5.7, 45.8, -86.0,
    -5.7, 57.3, -86.0,
    +5.7, 57.3, -86.0
])

# Build CasADi model
cmodel = cpin.Model(model_pin)
cdata = cmodel.createData()

def create_symbolic_ik_function():
    """Create symbolic IK function matching original numerical process"""
    # Symbolic inputs
    targets_xyz = ca.SX.sym('targets_xyz', 4, 3)
    q_guess = ca.SX.sym('q_guess', 12, 1)
    
    # Initialize joint angles
    q = q_guess
    
    # Iterative IK for each foot (same order as original)
    for fid in frame_ids:
        foot_idx = frame_ids.index(fid)
        tgt = targets_xyz[foot_idx, :].T  # Target as column vector
        
        for _ in range(IK_ITERS_PIN):
            # Forward kinematics
            cpin.forwardKinematics(cmodel, cdata, q)
            cpin.updateFramePlacements(cmodel, cdata)
            
            # Get current foot position
            p_cur = cdata.oMf[fid].translation
            
            # Compute error
            err = tgt - p_cur
            
            # Compute Jacobian (LOCAL_WORLD_ALIGNED)
            J6 = cpin.computeFrameJacobian(
                cmodel, cdata, q, fid, 
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            J_pos = J6[:3, :]  # Position part
            
            # Damped least squares
            J_dls = J_pos.T @ ca.inv(J_pos @ J_pos.T + DAMPING_PIN * ca.SX.eye(3))
            q += J_dls @ err
    
    return ca.Function('symbolic_ik', 
                      [targets_xyz, q_guess], 
                      [q],
                      ['targets_xyz', 'q_guess'],
                      ['q_out'])

def cubic_bezier_interpolation(z_start, z_end, t):
    """Matches original implementation exactly"""
    t_clamped = ca.fmax(0, ca.fmin(1, t))
    z_diff = z_end - z_start
    bezier_basis = t_clamped**3 + 3*(t_clamped**2)*(1 - t_clamped)
    return z_start + z_diff * bezier_basis

def create_one_step_function():
    """Create function for one timestep matching original logic"""
    # Inputs (all same as original per-timestep calculations)
    phase = ca.SX.sym('phase')
    p_foot_0 = ca.SX.sym('p_foot_0', 4, 3)
    p_foot_1 = ca.SX.sym('p_foot_1', 4, 3)
    x_i = ca.SX.sym('x_i')
    y_i = ca.SX.sym('y_i')
    theta_i = ca.SX.sym('theta_i')
    swing_height = ca.SX.sym('swing_height')
    
    # Cubic Bézier for world positions (matches original)
    foot_pos_world = cubic_bezier_interpolation(p_foot_0, p_foot_1, phase)
    
    # Body rotation matrix (matches original)
    c = ca.cos(theta_i)
    s = ca.sin(theta_i)
    R_body = ca.vertcat(
        ca.horzcat(c, s, 0),
        ca.horzcat(-s, c, 0),
        ca.horzcat(0, 0, 1)
    )
    
    # Body position
    p_body = ca.vertcat(x_i, y_i, 0)
    
    # Swing height calculation (matches original)
    if_phase = ca.if_else(phase <= 0.5, 
                         cubic_bezier_interpolation(0, swing_height, 2 * phase),
                         cubic_bezier_interpolation(swing_height, 0, 2 * phase - 1))
    z_i = if_phase
    
    # Foot positions in body frame (matches original)
    foot_pos_body = ca.SX(4, 3)
    for i in range(4):
        # World position of foot
        pw = foot_pos_world[i, :].T
        
        # Vector from COM to foot
        rel = pw - p_body
        
        # Rotate to body frame
        pb = R_body @ rel
        
        # Apply swing height
        foot_pos_body[i, :] = ca.vertcat(pb[0], pb[1], pb[2] + z_i).T
    
    # Create IK function
    ik_fn = create_symbolic_ik_function()
    
    # Call IK with body frame positions (matches original)
    q_out = ik_fn(targets_xyz=foot_pos_body, q_guess=ca.SX(q_init_np))['q_out']
    
    # Create function
    return ca.Function(
        'one_step_ref',
        [phase, p_foot_0, p_foot_1, x_i, y_i, theta_i, swing_height],
        [q_out, foot_pos_body],
        ['phase', 'p_foot_0', 'p_foot_1', 'x_i', 'y_i', 'theta_i', 'swing_height'],
        ['q_out', 'foot_pos_body']
    )

if __name__ == "__main__":
    # Create and save function
    fn = create_one_step_function()
    output_path = os.path.join(OUTPUT_DIR, "go2_one_step_ref.casadi")
    fn.save(output_path)
    print(f"Saved CasADi function to {output_path}")