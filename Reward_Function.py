import numpy as np

def compute_reward(r_current, r_target, n_F, theta_F, omega, omega_prev, torque):
  
    # 1. Accuracy: reward for pointing towards target
    angle_error = np.arccos(np.clip(np.dot(r_current, r_target), -1.0, 1.0))
    r_accuracy = -angle_error


    # 2. Safety: penalty if inside forbidden zone
    angle_F = np.arccos(np.clip(np.dot(r_current, n_F), -1.0, 1.0))
    
    theta_margin = angle_F - theta_F
    if theta_margin <= 0:
      r_safety = -100.0  # Big penalty if violated
    elif theta_margin < 0.1:  # Close to boundary (0.1 rad ≈ 6°)
        # Penalty increases as you get closer
      r_safety = -10.0 * (1.0 - theta_margin / 0.1)
    else:
      r_safety = 0.0  # Safe, no penalty

    # 3. Smoothness: penalize abrupt motion. Omega is angular velocity.
    delta_omega = omega - omega_prev
    r_smooth = -0.01 * np.linalg.norm(delta_omega)

    # 4. Efficiency: penalize high torque usage.
    r_efficiency = -0.05 * np.linalg.norm(torque) / 2.0

    # Total reward
    total_reward = r_accuracy + r_safety + r_smooth + r_efficiency

    return total_reward

# # Example vectors
# r_current = np.array([1, 0, 0])
# r_target  = np.array([0.866, 0.5, 0])
# n_F       = np.array([0, 1, 0])
# theta_F   = np.deg2rad(30)  # 30 degrees
# omega     = np.array([0.1, 0.2, 0])
# omega_prev= np.array([0.05, 0.1, 0])
# torque    = np.array([0.01, 0.02, 0.0])

# reward = compute_reward(r_current, r_target, n_F, theta_F, omega, omega_prev, torque)
# print("Reward:", reward)
