import matplotlib.pyplot as plt
import matplotlib
import os
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import SAC
import numpy as np
import imageio.v2 as imageio

from test_environment import SatDynEnv, scale_torque


def load_agent(model_name: str):
    """
    Load the agent to visualize.
    Inputs:
        model_name: Name of the model file (without .zip extension).
    Returns:
        model: The loaded SAC model.
    """
    model_path = f"models/{model_name}.zip"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = SAC.load(model_path)
    return model


def create_evaluation_env(initial_state):
    """
    Create the evaluation environment.
    """
    eval_env = SatDynEnv(render_mode="rgb_array", initial_state=initial_state)
    return eval_env


def print_rewards(model, eval_env, n_eval_episodes=10):
    """
    Evaluate and print the mean and std reward of the model.
    Inputs:
        model: The trained model.
        eval_env: The evaluation environment.
        n_eval_episodes: Number of episodes to evaluate.
    """
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


def simulate_agent(model: SAC, eval_env: SatDynEnv):
    """
    Simulate the agent in the evaluation environment.
    Inputs:
        model: The trained model.
        eval_env: The evaluation environment.
    """
    # Arrays for storing data
    times = np.linspace(0, 100, 100 * 10)
    states = []
    torques = []
    rewards = []
    frames = []

    kp = 50
    kd = 500

    obs, _ = eval_env.reset()
    done = False

    # Simulation loop
    while not done:
        action, _states = model.predict(obs, deterministic=True)

        states.append(obs.copy())
        torques.append(action.copy())

        # Step the environment
        obs, reward, done, truncated, info = eval_env.step(action)

        # Store reward
        rewards.append(reward)
        
        # Render the environment
        frame = eval_env.render()

        # Store frame
        frames.append(frame)

    eval_env.close()

    # Save as MP4
    output_path = "trajectory.mp4"

    imageio.mimsave(output_path, frames, fps=30)
    print(f"Saved video to {output_path}")

    # Extract the solution for attitude (in terms of quaternion) and angular velocity
    states_array = np.array(states)
    torques_array = np.array(torques) * scale_torque
    rewards_array = np.array(rewards)

    # Calculate cumulative reward
    cumulative_rewards = np.cumsum(rewards_array)

    # store the norm of quaternions
    norm_q = np.linalg.norm(states_array[:, :4], axis=1)

    simulation_data = {
        "quaternion": states_array[:, :4],
        "quaternion_norm": norm_q,
        "torques": torques_array,
        "omega": states_array[:, 4:7],
        "rewards": rewards_array,
        "cumulative_rewards": cumulative_rewards,
        "times": times
        }
    
    return simulation_data

def print_result(phi_final, omega_final, cumulative_reward_final):

    print(f"Final rotation angle φ: {phi_final:.6f}°")
    print(f"Final angular velocity: {np.sqrt(omega_final[0]**2 + omega_final[1]**2 + omega_final[2]**2)*180/np.pi:.6f} deg/s")
    print(f"Total cumulative reward: {cumulative_reward_final:.2f}")
    print(f"Target accuracy: 2.0°")
    print(f"Attitude control: {'✓ SUCCESS' if phi_final < 2.0 else '✗ NOT CONVERGED'}")
    print(f"Velocity settling: {'✓ SUCCESS' if np.sqrt(omega_final[0]**2 + omega_final[1]**2 + omega_final[2]**2)*180/np.pi < 0.5 else '✗ NOT SETTLED'}")

def plot_actual_attitude(simulation_data: dict):
    """Plot the ACTUAL satellite attitude trajectory based on rotation axis and angle phi"""
    
    # Switch to interactive backend for 3D plots
    matplotlib.use("TkAgg")

    # Parse quaternion and angular velocity components
    q_0, q_1, q_2, q_3 = simulation_data["quaternion"][:, 0], simulation_data["quaternion"][:, 1], simulation_data["quaternion"][:, 2], simulation_data["quaternion"][:, 3]
    omega_x, omega_y, omega_z = simulation_data["omega"][:, 0], simulation_data["omega"][:, 1], simulation_data["omega"][:, 2]
    norm_q = simulation_data["quaternion_norm"]
    torques_array = simulation_data["torques"]
    rewards_array = simulation_data["rewards"]
    cumulative_rewards = simulation_data["cumulative_rewards"]
    times = simulation_data["times"]

    def quat_to_axis_angle(q):
        """Convert quaternion to rotation axis and angle"""
        q0, q1, q2, q3 = q
        
        # Angle phi = 2 * arccos(|q0|) (same as in reward function)
        angle = 2 * np.arccos(np.abs(q0))
        
        # Rotation axis = q_vec / |q_vec| (normalized quaternion vector part)
        q_vec_norm = np.sqrt(q1**2 + q2**2 + q3**2)
        if q_vec_norm > 1e-6:  # Avoid division by zero
            axis = np.array([q1, q2, q3]) / q_vec_norm
        else:
            axis = np.array([0, 0, 0])  # No rotation axis if angle is zero
            
        return axis, angle
    
    # Extract rotation axes and angles for all time points
    rotation_axes = []
    rotation_angles = []
    
    for i in range(len(q_0)):
        axis, angle = quat_to_axis_angle([q_0[i], q_1[i], q_2[i], q_3[i]])
        rotation_axes.append(axis)
        rotation_angles.append(angle)
    
    # Convert to numpy arrays
    rotation_axes = np.array(rotation_axes)  # Shape: (N, 3)
    rotation_angles = np.array(rotation_angles)  # Shape: (N,)
    
    # Convert to degrees
    rotation_angles_deg = rotation_angles * 180 / np.pi
    
    #axis_x = rotation_axes[:, 0]
    #axis_y = rotation_axes[:, 1] 
    #axis_z = rotation_axes[:, 2]

    body_axis_arr = []
    for i in range(len(q_0)):
        q = [q_0[i], q_1[i], q_2[i], q_3[i]]
        w, x, y, z = q
        R = np.array([
                    [1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)],
                    [2*(x*y + z*w),         1 - 2*(x*x + z*z),   2*(y*z - x*w)],
                    [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x*x + y*y)]
                ])
        body_axis = R @ np.array([1, 0, 0])  # body X-axis
        body_axis_arr.append(body_axis)
    
    # Convert to numpy array for proper indexing
    body_axis_arr = np.array(body_axis_arr)
    
    fig = plt.figure(figsize=(18, 12))  # Larger figure for 6 subplots
    
    # 3D Rotation Axis Trajectory (This is the key trajectory for phi angle!)
    ax1 = fig.add_subplot(231, projection='3d')  # Changed to 2x3 grid
    
    # Plot trajectory on unit sphere (rotation axes are unit vectors)
    ax1.plot(body_axis_arr[:, 0], body_axis_arr[:, 1], body_axis_arr[:, 2], 'b-', alpha=0.7, linewidth=3, label='Boresight Axis Trajectory')
    ax1.scatter(body_axis_arr[0, 0], body_axis_arr[0, 1], body_axis_arr[0, 2], color='green', s=100, label='Start')
    ax1.scatter(body_axis_arr[-1, 0], body_axis_arr[-1, 1], body_axis_arr[-1, 2], color='red', s=100, label='End')
    ax1.scatter(1, 0, 0, color='gold', s=150, marker='*', label='Target')
    
    # Draw unit sphere wireframe
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_wireframe(sphere_x, sphere_y, sphere_z, alpha=0.1, color='gray')
    
    ax1.set_xlim([-1.1, 1.1])
    ax1.set_ylim([-1.1, 1.1])
    ax1.set_zlim([-1.1, 1.1])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Boresight Trajectory on Unit Sphere')
    ax1.legend()
    
    # Rotation angle φ vs time (same as in reward function)
    ax2 = fig.add_subplot(232)  # Changed to 2x3 grid
    ax2.plot(times[:len(rotation_angles_deg)], rotation_angles_deg, 'purple', linewidth=3, label='Angle φ')
    ax2.axhline(y=2.0, color='r', linestyle='--', linewidth=2, label='Target (2.0°)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Rotation Angle φ (°)')
    ax2.set_title('Rotation Angle φ vs Time\n(Used in reward function)')
    ax2.grid(True)
    ax2.legend()
    ax2.set_yscale('log')
    
    # Cumulative Reward vs Time
    ax5 = fig.add_subplot(233)  # New subplot for cumulative reward
    ax5.plot(times[:len(cumulative_rewards)], cumulative_rewards, 'orange', linewidth=3, label='Cumulative Reward')
    ax5.plot(times[:len(rewards_array)], rewards_array, 'lightcoral', alpha=0.6, linewidth=1, label='Step Reward')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Reward')
    ax5.set_title('Reward Evolution')
    ax5.grid(True)
    ax5.legend()
    
    # Plot quaternion
    plt.subplot(3, 2, 4)
    plt.plot(times, q_0, label='$q_0$')
    plt.plot(times, q_1, label='$q_1$')
    plt.plot(times, q_2, label='$q_2$')
    plt.plot(times, q_3, label='$q_3$')
    plt.plot(times, norm_q, label='norm')

    plt.title('Attitude')
    plt.ylabel('Quaternion')
    plt.legend()
    plt.grid()

    # Plot angular velocity
    plt.subplot(3, 2, 5)
    plt.plot(times, omega_x * (180 / np.pi), label='$\\omega_x$')
    plt.plot(times, omega_y * (180 / np.pi), label='$\\omega_y$')
    plt.plot(times, omega_z * (180 / np.pi), label='$\\omega_z$')
    plt.title('Angular velocity')
    plt.ylabel('$\\omega$ (deg/s)')
    plt.legend()
    plt.grid()

    # Plot torque input
    plt.subplot(3, 2, 6)
    plt.plot(times, torques_array[:, 0], label='$\\tau_x$')
    plt.plot(times, torques_array[:, 1], label='$\\tau_y$')
    plt.plot(times, torques_array[:, 2], label='$\\tau_z$')
    plt.title('Control torques')
    plt.xlabel('Time (s)')
    plt.ylabel('$\\tau$ (Nm)')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.show()

    print_result(rotation_angles_deg[-1], simulation_data["omega"][-1], cumulative_rewards[-1])
    
    return 
    

### MAIN ###
if __name__ == "__main__":
    model_name = "sac_sat_faster_2_latest"
    model = load_agent(model_name)

    # Set initial state for evaluation environment
    initial_state = [0.0, 90.0, 0.0, 0.1]  # [min_initial_angle, max_initial_angle, min_initial_angular_velocity, max_initial_angular_velocity]
    eval_env = create_evaluation_env(initial_state)

    print_rewards(model, eval_env, n_eval_episodes=10)
    simulation_data = simulate_agent(model, eval_env)
    plot_actual_attitude(simulation_data)