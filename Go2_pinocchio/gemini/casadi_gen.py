# create_casadi_function.py

import casadi as ca
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin # For CasADi interface to Pinocchio

# Pinocchio Model & constants
GO2_URDF = "Go2_pinocchio/go2_original.urdf" # ADJUST THIS PATH if necessary
FOOT_FRAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
DAMPING_PIN_CONST = 1e-4
IK_ITERS_PIN_CONST = 20 # Fixed iterations for SX-compatibility in IK loop

# Build Pinocchio model from URDF - this is the original, non-symbolic model
try:
    model_pin_orig = pin.buildModelFromUrdf(GO2_URDF)
except Exception as e:
    print(f"Error loading URDF: {e}")
    print(f"Please ensure '{GO2_URDF}' is the correct path to your URDF file.")
    exit()

frame_ids_const = [model_pin_orig.getFrameId(name) for name in FOOT_FRAMES]
nq_const = model_pin_orig.nq

# Helper: CasADi SX Cubic Bézier interpolation
def cubic_bezier_interpolation_sx(z_start_sx, z_end_sx, t_sx):
    t_clamped_sx = ca.fmax(0, ca.fmin(1, t_sx))
    z_diff_sx = z_end_sx - z_start_sx
    bezier_basis_sx = 3 * (t_clamped_sx**2) - 2 * (t_clamped_sx**3)
    return z_start_sx + z_diff_sx * bezier_basis_sx

# Helper: CasADi SX Inverse Kinematics using Pinocchio.casadi
def inverse_kinematics_casadi_sx(
    targets_xyz_sx,      # SX (4,3), target foot positions in body frame
    foot_frame_ids_list, # Python list of int frame IDs
    pin_model_sx,        # <<< Pinocchio model CAST for ca.SX >>>
    q_guess_sx,          # SX (nq,), initial guess for joint angles
    max_iter_const,      # Python int, number of iterations
    damping_const        # Python float, damping factor
):
    q_sx = ca.SX(q_guess_sx)
    # Data must be created from the SX-casted model
    cas_data_sx = pin_model_sx.createData() # <<< Data for SX model >>>

    for k_foot_idx, frame_id in enumerate(foot_frame_ids_list):
        target_pos_sx = targets_xyz_sx[k_foot_idx, :].T  # SX (3,1)

        for _ in range(max_iter_const):
            # Use the SX-casted model and its corresponding data object
            cpin.forwardKinematics(pin_model_sx, cas_data_sx, q_sx)
            cpin.updateFramePlacements(pin_model_sx, cas_data_sx)
            current_pos_sx = cas_data_sx.oMf[frame_id].translation
            err_sx = target_pos_sx - current_pos_sx
            
            J6_sx = cpin.computeFrameJacobian(pin_model_sx, cas_data_sx, q_sx, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            J_pos_sx = J6_sx[:3, :]

            A_sx = J_pos_sx @ J_pos_sx.T + damping_const * ca.SX.eye(3)
            delta_q_sx = J_pos_sx.T @ ca.solve(A_sx, err_sx)
            q_sx = q_sx + delta_q_sx
    return q_sx

def create_go2_trajectory_step_function():
    # Symbolic inputs varying per time step
    t_i_sx = ca.SX.sym("t_i_sx")
    x_t_i_sx = ca.SX.sym("x_t_i_sx")
    y_t_i_sx = ca.SX.sym("y_t_i_sx")
    theta_t_i_sx = ca.SX.sym("theta_t_i_sx")

    # Symbolic inputs (parameters) fixed over N steps for one trajectory
    T_param_sx = ca.SX.sym("T_param_sx")
    swing_height_param_sx = ca.SX.sym("swing_height_param_sx")
    p_foot_0_world_sx = ca.SX.sym("p_foot_0_world_sx", 4, 3)
    p_foot_1_world_sx = ca.SX.sym("p_foot_1_world_sx", 4, 3)
    q_ik_guess_param_sx = ca.SX.sym("q_ik_guess_param_sx", nq_const)

    # <<< Cast the original model to its CasADi SX representation ONCE >>>
    # This sx_model will be used in all cpin calls via inverse_kinematics_casadi_sx
    model_pin_sx = model_pin_orig.cast(ca.SX)

    # 1. Phase logic
    phase_raw_sx = (t_i_sx / T_param_sx - 0.25) * 2.0
    phase_sx = ca.fmax(0.0, ca.fmin(1.0, phase_raw_sx))

    # 2. Swing height (z_i)
    phase_norm_up_sx = phase_sx * 2.0
    phase_norm_down_sx = (phase_sx - 0.5) * 2.0
    z_lift_up_sx = cubic_bezier_interpolation_sx(ca.SX(0.0), swing_height_param_sx, phase_norm_up_sx)
    z_lift_down_sx = cubic_bezier_interpolation_sx(swing_height_param_sx, ca.SX(0.0), phase_norm_down_sx)
    z_i_lift_sx = ca.if_else(phase_sx <= 0.5, z_lift_up_sx, z_lift_down_sx)

    # 3. Interpolate foot EEs' target position in WORLD frame
    foot_pos_world_target_sx = cubic_bezier_interpolation_sx(p_foot_0_world_sx, p_foot_1_world_sx, phase_sx)

    # 4. Body rotation matrix (world to body)
    c_theta_sx = ca.cos(theta_t_i_sx); s_theta_sx = ca.sin(theta_t_i_sx)
    body_rot_sx = ca.vertcat(
        ca.horzcat(c_theta_sx, s_theta_sx, 0.0),
        ca.horzcat(-s_theta_sx, c_theta_sx, 0.0),
        ca.horzcat(0.0, 0.0, 1.0)
    )

    # 5. Transform world targets to body frame, add swing height
    com_pos_world_i_sx = ca.vertcat(x_t_i_sx, y_t_i_sx, 0.0)
    foot_pos_body_ik_target_list_sx = []
    for f_idx in range(4):
        p_world_f_sx = foot_pos_world_target_sx[f_idx, :].T
        vec_com_to_foot_world_sx = p_world_f_sx - com_pos_world_i_sx
        p_body_no_lift_sx = body_rot_sx @ vec_com_to_foot_world_sx
        foot_target_body_f_sx = ca.vertcat(
            p_body_no_lift_sx[0],
            p_body_no_lift_sx[1],
            p_body_no_lift_sx[2] + z_i_lift_sx
        )
        foot_pos_body_ik_target_list_sx.append(foot_target_body_f_sx)
    foot_pos_body_ik_target_sx = ca.horzcat(*foot_pos_body_ik_target_list_sx).T

    # 6. CasADi Inverse Kinematics
    q_solved_sx = inverse_kinematics_casadi_sx(
        targets_xyz_sx=foot_pos_body_ik_target_sx,
        foot_frame_ids_list=frame_ids_const,
        pin_model_sx=model_pin_sx, # <<< Pass the SX-casted model >>>
        q_guess_sx=q_ik_guess_param_sx,
        max_iter_const=IK_ITERS_PIN_CONST,
        damping_const=DAMPING_PIN_CONST
    )

    # Define CasADi Function
    map_iterated_inputs = [t_i_sx, x_t_i_sx, y_t_i_sx, theta_t_i_sx]
    map_iterated_input_names = ["t_curr", "com_x_curr", "com_y_curr", "com_theta_curr"]
    map_fixed_inputs = [
        T_param_sx, swing_height_param_sx,
        p_foot_0_world_sx, p_foot_1_world_sx, q_ik_guess_param_sx
    ]
    map_fixed_input_names = [
        "T_horizon", "swing_h_param",
        "p_foot_0_w", "p_foot_1_w", "q_ik_guess"
    ]
    all_inputs = map_iterated_inputs + map_fixed_inputs
    all_input_names = map_iterated_input_names + map_fixed_input_names
    outputs = [q_solved_sx, foot_pos_body_ik_target_sx]
    output_names = ["q_out", "foot_pos_body_target_out"]

    casadi_fn = ca.Function(
        "go2_trajectory_step_casadi", all_inputs, outputs,
        all_input_names, output_names
    )
    return casadi_fn

if __name__ == "__main__":
    go2_step_func = create_go2_trajectory_step_function()
    print(f"CasADi function created: {go2_step_func.name()}")
    file_name = "go2_trajectory_step.casadi"
    go2_step_func.save(file_name)
    print(f"CasADi function saved to '{file_name}'")
    print(f"Number of inputs: {go2_step_func.n_in()}, Names: {go2_step_func.name_in()}")
    print(f"Number of outputs: {go2_step_func.n_out()}, Names: {go2_step_func.name_out()}")