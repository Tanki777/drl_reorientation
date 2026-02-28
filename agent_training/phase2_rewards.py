import numpy as np
import math

""" phase1_best1_3cont """
def reward_function(state, scale_torque_norm, episode_count, use_curr_koz_weight):
    q0_current = state[0]
    ang_vel_sat_1 = state[4]
    ang_vel_sat_2 = state[5]
    ang_vel_sat_3 = state[6]
    q0_prev = state[10]  
    torque_1 = state[14]
    torque_2 = state[15]
    torque_3 = state[16]
    margin_koz = state[20]
    scale_torque_norm = np.float32(scale_torque_norm)

    margin_safe = 0.05 # 3 deg in rad
    violation_coeff = 10.0
    buffer_coeff = 0.2
    koz_weight = min(1.0, episode_count / 20.0)
    
    # Clamp q0 values to [-1, 1] to prevent acos() domain errors (NaN) with large torques
    # Using min/max instead of np.clip for numba compatibility with scalars
    q0_current = min(max(q0_current, -1.0), 1.0)
    q0_prev = min(max(q0_prev, -1.0), 1.0)
    
    err_phi_current = 2 * math.acos(q0_current)   # in [rad]
    err_phi_prev = 2 * math.acos(q0_prev)   # in [rad]

    err_phi_current = err_phi_current * 180.0 / np.pi
    err_phi_prev = err_phi_prev * 180.0 / np.pi

    safety_factor = min(1.0, margin_koz / margin_safe)
    safety_factor = max(0.0, safety_factor)

    # Reward for reducing attitude error
    r1 = (err_phi_prev - err_phi_current) * safety_factor  # positive if error decreased

    r2 = 0.0
    if np.sqrt(ang_vel_sat_1**2 + ang_vel_sat_2**2 + ang_vel_sat_3**2) >= 0.1745:  # 10 deg/s in rad/s
        r2 = -10.0  # small penalty for high angular velocity

    # Bonus for high accuracy
    r3 = 0.0
    if err_phi_current <= 0.25:
        r3 = 0.05  # bonus for reaching the goal
    else:
        r3 = -0.05

    # Penalty for using large torques
    r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

    # Penalty for entering / being close to keep out zone
    r5 = 0.0
    if margin_koz <= 0.0:
        r5 = -1.0
    # elif margin_koz < margin_safe:
    #     r5 = -buffer_coeff * (margin_safe - margin_koz) / margin_safe

    if use_curr_koz_weight:
        r5 = r5 * koz_weight
    

    return r1 + r2 + r3 + r4 + r5

""" phase1_best1_3 """
def reward_function(state, scale_torque_norm, episode_count, use_curr_koz_weight):
    q0_current = state[0]
    ang_vel_sat_1 = state[4]
    ang_vel_sat_2 = state[5]
    ang_vel_sat_3 = state[6]
    q0_prev = state[10]  
    torque_1 = state[14]
    torque_2 = state[15]
    torque_3 = state[16]
    margin_koz = state[20]
    scale_torque_norm = np.float32(scale_torque_norm)

    margin_safe = 0.05 # 3 deg in rad
    violation_coeff = 10.0
    buffer_coeff = 0.2
    koz_weight = min(1.0, episode_count / 20.0)
    
    # Clamp q0 values to [-1, 1] to prevent acos() domain errors (NaN) with large torques
    # Using min/max instead of np.clip for numba compatibility with scalars
    q0_current = min(max(q0_current, -1.0), 1.0)
    q0_prev = min(max(q0_prev, -1.0), 1.0)
    
    err_phi_current = 2 * math.acos(q0_current)   # in [rad]
    err_phi_prev = 2 * math.acos(q0_prev)   # in [rad]

    err_phi_current = err_phi_current * 180.0 / np.pi
    err_phi_prev = err_phi_prev * 180.0 / np.pi

    safety_factor = min(1.0, margin_koz / margin_safe)
    safety_factor = max(0.0, safety_factor)

    # Reward for reducing attitude error
    r1 = (err_phi_prev - err_phi_current)  # positive if error decreased

    r2 = 0.0
    if np.sqrt(ang_vel_sat_1**2 + ang_vel_sat_2**2 + ang_vel_sat_3**2) >= 0.1745:  # 10 deg/s in rad/s
        r2 = -10.0  # small penalty for high angular velocity

    # Bonus for high accuracy
    r3 = 0.0
    if err_phi_current <= 0.25:
        r3 = 0.01  # bonus for reaching the goal
    else:
        r3 = -0.01

    # Penalty for using large torques
    r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

    # Penalty for entering / being close to keep out zone
    r5 = 0.0
    if margin_koz <= 0.0:
        r5 = -1.0
    else:
        r5 = -1*math.exp(-66.0*margin_koz)

    

    return r1 + r2 + r3 + r4 + r5


""" phase1_best1_3cont2 """
def reward_function(state, scale_torque_norm, episode_count, use_curr_koz_weight):
    q0_current = state[0]
    ang_vel_sat_1 = state[4]
    ang_vel_sat_2 = state[5]
    ang_vel_sat_3 = state[6]
    q0_prev = state[10]  
    torque_1 = state[14]
    torque_2 = state[15]
    torque_3 = state[16]
    margin_koz = state[20]
    scale_torque_norm = np.float32(scale_torque_norm)

    margin_safe = 0.05 # 3 deg in rad
    violation_coeff = 10.0
    buffer_coeff = 0.2
    koz_weight = min(1.0, episode_count / 20.0)
    
    # Clamp q0 values to [-1, 1] to prevent acos() domain errors (NaN) with large torques
    # Using min/max instead of np.clip for numba compatibility with scalars
    q0_current = min(max(q0_current, -1.0), 1.0)
    q0_prev = min(max(q0_prev, -1.0), 1.0)
    
    err_phi_current = 2 * math.acos(q0_current)   # in [rad]
    err_phi_prev = 2 * math.acos(q0_prev)   # in [rad]

    err_phi_current = err_phi_current * 180.0 / np.pi
    err_phi_prev = err_phi_prev * 180.0 / np.pi

    safety_factor = min(1.0, margin_koz / margin_safe)
    safety_factor = max(0.0, safety_factor)

    # Reward for reducing attitude error
    r1 = (err_phi_prev - err_phi_current)  # positive if error decreased

    r2 = 0.0
    if np.sqrt(ang_vel_sat_1**2 + ang_vel_sat_2**2 + ang_vel_sat_3**2) >= 0.1745:  # 10 deg/s in rad/s
        r2 = -1.0  # small penalty for high angular velocity

    # Bonus for high accuracy
    r3 = 0.0
    if err_phi_current <= 0.25:
        r3 = 0.05  # bonus for reaching the goal
    else:
        r3 = -0.01

    # Penalty for using large torques
    r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

    # Penalty for entering / being close to keep out zone
    r5 = 0.0
    if margin_koz <= 0.0:
        r5 = -1.0
    elif margin_koz < margin_safe:
        r5 = -1*(margin_safe - margin_koz)/margin_safe

    return r1 + r2 + r3 + r4 + r5