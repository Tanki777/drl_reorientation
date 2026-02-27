import os
import matplotlib.pyplot as plt
import matplotlib
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import SAC
import numpy as np
import imageio.v2 as imageio
import csv
import time

from environment import SatDynEnv, scale_torque, scale_angular_velocity_sat, scale_angular_velocity_wheels, scale_margin_koz
from constants import dt


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


def create_evaluation_env(initial_state, use_safety_filter):
    """
    Create the evaluation environment.
    """
    eval_env = SatDynEnv(render_mode="rgb_array", initial_state=initial_state, use_safety_filter=use_safety_filter)
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


def simulate_agent(model: SAC, eval_env: SatDynEnv, max_steps: int):
    """
    Simulate the agent in the evaluation environment.
    Inputs:
        model: The trained model.
        eval_env: The evaluation environment.
    """
    # Arrays for storing data
    times = np.linspace(0, max_steps/10, max_steps)  # Assuming dt=0.1s
    states = []
    torques = []
    rewards = []
    frames = []

    kp = 50
    kd = 500

    obs, _ = eval_env.reset()
    done = False
    normal_vector_koz = eval_env.normal_vector_koz
    half_angle_koz = eval_env.half_angle_koz
    min_margin_koz = 0
    cnt_Koz_violations = 0

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
        #frame = eval_env.render()

        # Store frame
        #frames.append(frame)

        min_margin_koz = eval_env.min_margin_koz
        cnt_Koz_violations = eval_env.entered_koz_count

    eval_env.close()

    # Save as MP4
    output_path = "trajectory.mp4"

    #imageio.mimsave(output_path, frames, fps=30)
    #print(f"Saved video to {output_path}")

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
        "omega": states_array[:, 4:7]*scale_angular_velocity_sat,
        "rewards": rewards_array,
        "cumulative_rewards": cumulative_rewards,
        "times": times,
        "normal_vector_koz": normal_vector_koz,
        "half_angle_koz": half_angle_koz,
        "margin_angles_koz": states_array[:, 20]*scale_margin_koz*180/np.pi,
        "min_margin_koz": min_margin_koz,
        "cnt_Koz_violations": cnt_Koz_violations
        }
    
    return simulation_data

def evaluate_agent_worker(model_name: str, initial_state: list, use_safety_filter: int, max_steps: int, episodes: int, worker_id: int):
    """
    Worker function to evaluate agent for a subset of episodes.
    This function will be run in parallel by multiple processes.
    Inputs:
        model_name: Name of the model file (without .zip extension).
        initial_state: Initial state configuration for environment.
        use_safety_filter: Safety filter mode.
        max_steps: Maximum steps per episode.
        episodes: Number of episodes to run.
        worker_id: ID of this worker process.
    Returns:
        Dictionary containing evaluation results.
    """
    # Load model in worker process
    model = load_agent(model_name)
    
    # Create environment in worker process
    eval_env = create_evaluation_env(initial_state, use_safety_filter)
    
    koz_violation_episodes = 0
    ep_rewards = []
    err_angles_final = []
    ang_vels_final = []
    min_margins_koz = []
    cnts_koz_violations = []
    simulation_data = []

    for episode in range(episodes):
        print(f"Worker {worker_id}: Episode {episode+1}/{episodes}...", end="\r")
        times = np.linspace(0, max_steps/10, max_steps)  # Assuming dt=0.1s
        states = []
        torques = []
        rewards = []

        obs, _ = eval_env.reset()
        done = False
        reward_cum = 0 # for this episode

        # Simulation loop
        while not done:
            action_agent, _states = model.predict(obs, deterministic=True)

            states.append(obs.copy())
            #torques.append(action_agent.copy())

            # Step the environment
            obs, reward, done, truncated, info, action_filtered = eval_env.step(action_agent) # action_filtered is the filter output if applied, else same as action_agent
            torques.append(action_filtered.copy())
            rewards.append(reward)

            # Add up reward
            reward_cum += reward

        if eval_env.entered_koz_count > 0:
            koz_violation_episodes += 1

        # Convert to numpy arrays
        states_array = np.array(states)
        torques_array = np.array(torques) * scale_torque
        rewards_array = np.array(rewards)

        # Store episode data in a dictionary
        episode_data = {
            "quaternion": states_array[:, :4],
            "quaternion_norm": np.linalg.norm(states_array[:, :4], axis=1),
            "torques": torques_array,
            "omega": states_array[:, 4:7]*scale_angular_velocity_sat,
            "rewards": rewards_array,
            "cumulative_rewards": np.cumsum(rewards_array),
            "times": times,
            "normal_vector_koz": eval_env.normal_vector_koz,
            "half_angle_koz": eval_env.half_angle_koz,
            "margin_angles_koz": states_array[:, 20]*scale_margin_koz*180/np.pi,
            "min_margin_koz": eval_env.min_margin_koz,
            "cnt_Koz_violations": eval_env.entered_koz_count,
            "filter_log": eval_env.filter_log
        }

        simulation_data.append(episode_data)
        
        # Add episode results to evaluation lists
        min_margins_koz.append(eval_env.min_margin_koz*180/np.pi)  # in degrees
        cnts_koz_violations.append(eval_env.entered_koz_count)
        ep_rewards.append(reward_cum)

        q0_final = obs[0]
        err_angle_final = 2 * np.arccos(np.abs(q0_final)) * 180/np.pi  # final rotation angle in degrees
        err_angles_final.append(err_angle_final)

        ang_vel_final = obs[4:7] * scale_angular_velocity_sat * 180/np.pi  # final angular velocity in deg/s
        ang_vel_final_mag = np.sqrt(ang_vel_final[0]**2 + ang_vel_final[1]**2 + ang_vel_final[2]**2) # magnitude
        ang_vels_final.append(ang_vel_final_mag)

    eval_env.close()
    
    # Return results from this worker
    return {
        "koz_violation_episodes": koz_violation_episodes,
        "ep_rewards": ep_rewards,
        "err_angles_final": err_angles_final,
        "ang_vels_final": ang_vels_final,
        "min_margins_koz": min_margins_koz,
        "cnts_koz_violations": cnts_koz_violations,
        "simulation_data": simulation_data
    }


def evaluate_agent(model_name: str, initial_state: list, use_safety_filter: int, max_steps: int, episodes: int, num_workers: int = 8):
    """
    Simulate the agent in parallel using multiple processes.
    Inputs:
        model_name: Name of the model file (without .zip extension).
        initial_state: Initial state configuration for environment.
        use_safety_filter: Safety filter mode.
        max_steps: Maximum steps per episode.
        episodes: Total number of episodes to run.
        num_workers: Number of parallel worker processes (default: 8).
    """
    import multiprocessing as mp
    
    timestamp = time.time()
    
    # Divide episodes among workers
    episodes_per_worker = episodes // num_workers
    remaining_episodes = episodes % num_workers
    
    # Create list of episode counts for each worker
    episode_counts = [episodes_per_worker] * num_workers
    for i in range(remaining_episodes):
        episode_counts[i] += 1
    
    print(f"Running {episodes} episodes across {num_workers} workers...")
    print(f"Episodes per worker: {episode_counts}")
    print()
    
    # Create pool of workers
    with mp.Pool(processes=num_workers) as pool:
        # Start all workers in parallel
        results = []
        for worker_id in range(num_workers):
            result = pool.apply_async(
                evaluate_agent_worker,
                args=(model_name, initial_state, use_safety_filter, max_steps, episode_counts[worker_id], worker_id)
            )
            results.append(result)
        
        # Wait for all workers to complete and collect results
        worker_results = [r.get() for r in results]
    
    # Combine results from all workers
    koz_violation_episodes = sum(r["koz_violation_episodes"] for r in worker_results)
    ep_rewards = []
    err_angles_final = []
    ang_vels_final = []
    min_margins_koz = []
    cnts_koz_violations = []
    simulation_data = []
    
    for r in worker_results:
        ep_rewards.extend(r["ep_rewards"])
        err_angles_final.extend(r["err_angles_final"])
        ang_vels_final.extend(r["ang_vels_final"])
        min_margins_koz.extend(r["min_margins_koz"])
        cnts_koz_violations.extend(r["cnts_koz_violations"])
        simulation_data.extend(r["simulation_data"])

    # Convert to numpy arrays
    err_angles_final_array = np.array(err_angles_final)
    ang_vels_final_array = np.array(ang_vels_final)
    min_margins_koz_array = np.array(min_margins_koz)
    cnts_koz_violations_array = np.array(cnts_koz_violations)

    # Save episode data (per timestep)
    np.savez(f"evaluation_{timestamp}.npz", data=np.array(simulation_data), dtype=object)

    # Print results
    print()
    print(f"Violation rate: {koz_violation_episodes/episodes*100}%")
    print(f"Agent was within KOZ for these amount of steps per episode:")
    print(f"--- Mean: {np.mean(cnts_koz_violations_array):.2f}, Std: {np.std(cnts_koz_violations_array):.2f}, Min: {np.min(cnts_koz_violations_array)}, Max: {np.max(cnts_koz_violations_array)}")
    print(f"Minimum margin to KOZ (degrees):")
    print(f"--- Mean: {np.mean(min_margins_koz_array):.2f}, Std: {np.std(min_margins_koz_array):.2f}, Min: {np.min(min_margins_koz_array):.2f}, Max: {np.max(min_margins_koz_array):.2f}")
    print(f"Final rotation angle (degrees):")
    print(f"--- Mean: {np.mean(err_angles_final_array):.2f}, Std: {np.std(err_angles_final_array):.2f}, Min: {np.min(err_angles_final_array):.2f}, Max: {np.max(err_angles_final_array):.2f}")
    print(f"Final angular velocity (deg/s):")
    print(f"--- Mean: {np.mean(ang_vels_final_array):.4f}, Std: {np.std(ang_vels_final_array):.4f}, Min: {np.min(ang_vels_final_array):.4f}, Max: {np.max(ang_vels_final_array):.4f}")
    print(f"Rewards:")
    print(f"--- Mean: {np.mean(ep_rewards):.2f}, Std: {np.std(ep_rewards):.2f}, Min: {np.min(ep_rewards):.2f}, Max: {np.max(ep_rewards):.2f}")


def print_result(phi_final, omega_final, cumulative_reward_final):

    print(f"Final rotation angle: {phi_final:.6f}°")
    print(f"Final angular velocity: {np.sqrt(omega_final[0]**2 + omega_final[1]**2 + omega_final[2]**2)*180/np.pi:.6f} deg/s")
    print(f"Total cumulative reward: {cumulative_reward_final:.2f}")
    print(f"Target accuracy: 2.0°")
    print(f"Attitude control: {"SUCCESS" if phi_final < 2.0 else "NOT CONVERGED"}")
    print(f"Velocity settling: {"SUCCESS" if np.sqrt(omega_final[0]**2 + omega_final[1]**2 + omega_final[2]**2)*180/np.pi < 0.5 else "NOT SETTLED"}")

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
    normal_vector_koz = simulation_data["normal_vector_koz"]
    half_angle_koz = simulation_data["half_angle_koz"]
    margin_angles_koz = simulation_data["margin_angles_koz"]
    min_margin_koz = simulation_data["min_margin_koz"]
    cnt_Koz_violations = simulation_data["cnt_Koz_violations"]

    print("Minimum margin KOZ:", min_margin_koz*180/np.pi, "degrees")
    print("Count KOZ violations:", cnt_Koz_violations)
    print("Half angle KOZ:", half_angle_koz*180/np.pi, "degrees")
    

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
    
    fig = plt.figure(figsize=(18, 12))
    
    # 3D Rotation Axis Trajectory (This is the key trajectory for phi angle!)
    ax1 = fig.add_subplot(241, projection="3d")
    
    # Plot trajectory on unit sphere (rotation axes are unit vectors)
    ax1.plot(body_axis_arr[:, 0], body_axis_arr[:, 1], body_axis_arr[:, 2], "b-", alpha=0.7, linewidth=3, label="Boresight Axis Trajectory")
    ax1.scatter(body_axis_arr[0, 0], body_axis_arr[0, 1], body_axis_arr[0, 2], color="green", s=100, label="Start")
    ax1.scatter(body_axis_arr[-1, 0], body_axis_arr[-1, 1], body_axis_arr[-1, 2], color="red", s=100, label="End")
    ax1.scatter(1, 0, 0, color="gold", s=150, marker="*", label="Target")

    def _generate_keep_out_zone_circle():
        # Create circle points for the keep out zone
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_points = []
        for angle in theta:
            # Generate points on the circle in the plane perpendicular to the normal vector
            v = np.array([np.cos(angle), np.sin(angle), 0])
            # Rotate v to be perpendicular to koz_normal
            if np.allclose(normal_vector_koz, [0, 0, 1]):
                rot_axis = np.array([1, 0, 0])
            else:
                rot_axis = np.cross([0, 0, 1], normal_vector_koz)
                rot_axis /= np.linalg.norm(rot_axis)
            angle_to_rotate = np.arccos(np.dot(normal_vector_koz, [0, 0, 1]))
            # Rodrigues' rotation formula
            v_rotated = (v * np.cos(angle_to_rotate) +
                        np.cross(rot_axis, v) * np.sin(angle_to_rotate) +
                        rot_axis * np.dot(rot_axis, v) * (1 - np.cos(angle_to_rotate)))
            # Scale to the radius of the keep out zone circle
            radius = np.sin(half_angle_koz)
            circle_point = normal_vector_koz * np.cos(half_angle_koz) + v_rotated * radius
            circle_points.append(circle_point)

        return circle_points

    # Plot keep out zone as a ring on the unit sphere
    if normal_vector_koz is not None and half_angle_koz is not None:
        circle_points = _generate_keep_out_zone_circle()
        circle_points = np.array(circle_points)
        ax1.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], "orange", linewidth=2, label="Keep Out Zone")
    
    # Draw unit sphere wireframe
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_wireframe(sphere_x, sphere_y, sphere_z, alpha=0.1, color="gray")
    
    ax1.set_xlim([-1.1, 1.1])
    ax1.set_ylim([-1.1, 1.1])
    ax1.set_zlim([-1.1, 1.1])
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("3D Boresight Trajectory on Unit Sphere")
    ax1.legend()
    
    # Rotation angle φ vs time (same as in reward function)
    ax2 = fig.add_subplot(242)
    ax2.plot(times[:len(rotation_angles_deg)], rotation_angles_deg, "purple", linewidth=3, label="Angle $\\phi$")
    ax2.axhline(y=0.25, color="r", linestyle="--", linewidth=2, label="Accuracy Threshold (0.25°)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Rotation Angle $\\phi$ (°)")
    ax2.set_title("Rotation Angle $\\phi$ vs Time\n(Used in reward function)")
    ax2.grid(True)
    ax2.legend()
    ax2.set_yscale("log")
    
    # Cumulative Reward vs Time
    ax3 = fig.add_subplot(243)  # New subplot for cumulative reward
    ax3.plot(times[:len(cumulative_rewards)], cumulative_rewards, "orange", linewidth=3, label="Cumulative Reward")
    ax3.plot(times[:len(rewards_array)], rewards_array, "lightcoral", alpha=0.6, linewidth=1, label="Step Reward")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Reward")
    ax3.set_title("Reward Evolution")
    ax3.grid(True)
    ax3.legend()
    
    # Plot quaternion
    ax4 = fig.add_subplot(244)
    ax4.plot(times, q_0, label="$q_0$")
    ax4.plot(times, q_1, label="$q_1$")
    ax4.plot(times, q_2, label="$q_2$")
    ax4.plot(times, q_3, label="$q_3$")
    ax4.plot(times, norm_q, label="norm")
    ax4.set_title("Attitude")
    ax4.set_ylabel("Quaternion")
    ax4.legend()
    ax4.grid()

    # Plot angular velocity
    ax5 = fig.add_subplot(245)
    ax5.plot(times, omega_x * (180 / np.pi), label="$\\omega_x$")
    ax5.plot(times, omega_y * (180 / np.pi), label="$\\omega_y$")
    ax5.plot(times, omega_z * (180 / np.pi), label="$\\omega_z$")
    ax5.set_title("Angular velocity")
    ax5.set_ylabel("$\\omega$ (deg/s)")
    ax5.legend()
    ax5.grid()
    # Plot torque input
    ax6 = fig.add_subplot(246)
    ax6.plot(times, torques_array[:, 0], label="$\\tau_1$")
    ax6.plot(times, torques_array[:, 1], label="$\\tau_2$")
    ax6.plot(times, torques_array[:, 2], label="$\\tau_3$")
    ax6.set_title("Control torques")
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("$\\tau$ (Nm)")
    ax6.legend()
    ax6.grid()

    # Plot keep out zone margin angle
    ax7 = fig.add_subplot(247)
    ax7.plot(times, margin_angles_koz, label="Margin Angle KOZ")
    ax7.set_title("Keep Out Zone Margin Angle")
    ax7.set_ylabel("Angle (degrees)")
    ax7.legend()
    ax7.grid()
    
    plt.tight_layout()
    plt.show()

    print("Initial rotation angle:", rotation_angles_deg[0], "degrees")

    print_result(rotation_angles_deg[-1], simulation_data["omega"][-1], cumulative_rewards[-1])

    # calculate norm for each vector in torque array
    torque_norms = np.linalg.norm(torques_array, axis=1)
    print("Torque avg: ", np.mean(torque_norms))
    # print(f"first 10 torque array elements: {torques_array[:10]}")
    # print(f"first 10 torque norms: {torque_norms[:10]}")
    # print(f"sum of norms: {np.sum(torque_norms**2)}")
    
    return 


def load_evaluation_data(file_path: str):
    """
    Load the simulation data.
    The part inside of the for loop can be modified to find specific episodes you are interested in.
    For example, episodes where the mean torque was above some threshold, episodes where the safety constraint was violated, etc.
    You could then print the index (i) of these episodes to the terminal.
    """
    _data = np.load(file_path, allow_pickle=True)
    data = _data["data"].tolist()  # Convert back to list of dicts

    reward_min = None
    reward_min_idx = None
    settled_count = 0
    settled_final_err = []

    for i, episode_data in enumerate(data):
        if np.mean(np.linalg.norm(episode_data["torques"], axis=1)) > 0.0004 and episode_data["cnt_Koz_violations"] > 0:
            #print(i,end=",")
            pass

        if 2 * np.arccos(np.abs(episode_data["quaternion"][-1,0])) * 180/np.pi < 1.0 and episode_data["cnt_Koz_violations"] > 0:
            #print(i,end=",")
            pass
        
        # Off-border: no violation, target reached, large KOZ
        if (2 * np.arccos(np.abs(episode_data["quaternion"][-1,0])) * 180/np.pi < 0.5 and episode_data["min_margin_koz"]*180/np.pi > 10.0 
            and 2 * np.arccos(np.abs(episode_data["quaternion"][0,0])) * 180/np.pi < 100.0 and episode_data["cnt_Koz_violations"] == 0 and episode_data["half_angle_koz"]*180/np.pi > 25.0):
            #print(i,end=",")
            pass

        # Along-border: no violation, target reached, large KOZ
        if (2 * np.arccos(np.abs(episode_data["quaternion"][-1,0])) * 180/np.pi < 0.5 and episode_data["min_margin_koz"]*180/np.pi < 0.5 
            and 2 * np.arccos(np.abs(episode_data["quaternion"][0,0])) * 180/np.pi < 100.0 and episode_data["cnt_Koz_violations"] == 0 and episode_data["half_angle_koz"]*180/np.pi > 25.0):
            #print(i,end=",")
            pass
        
        # Stuck: Low episode reward, low control effort, far away from target
        if (episode_data["cumulative_rewards"][-1] < -100 and np.sum(np.linalg.norm(episode_data["torques"], axis=1)**2) < 0.0005
            and 2 * np.arccos(np.abs(episode_data["quaternion"][-1,0])) * 180/np.pi > 30.0):
            #print(i,end=",")
            pass

        # Oscillation: Low episode reward, high control effort, close to target
        if (episode_data["cumulative_rewards"][-1] < -100 and np.sum(np.linalg.norm(episode_data["torques"], axis=1)**2) > 0.0008
            and 2 * np.arccos(np.abs(episode_data["quaternion"][-1,0])) * 180/np.pi < 10.0):
            #print(i,end=",")
            pass

        if 2 * np.arccos(np.abs(episode_data["quaternion"][-1,0])) * 180/np.pi > 30.0:
            #print(i,end=",")
            pass
        
        # Violation
        if episode_data["cnt_Koz_violations"] > 0:
            print(i,end=",")
            pass

        if episode_data["cumulative_rewards"][-1] < -2800:
            #print(f"{i}:\n{episode_data["filter_log"]}")
            pass

        if i == 0 and "filter_log" not in episode_data:
            print(f"{i} does not have filter log")

        if "filter_log" in episode_data and episode_data["filter_log"] != "":
            #print(f"{i}:\n{episode_data["filter_log"]}")
            pass

        if 2 * np.arccos(np.abs(episode_data["quaternion"][0,0])) * 180/np.pi < 120 and episode_data["cumulative_rewards"][-1] > 120:
            #print(i,end=",")
            pass

        if np.linalg.norm(episode_data["omega"][-1]) * 180 / np.pi > 2.0:
            #print(i,end=",")
            pass

        if np.mean(np.linalg.norm(episode_data["omega"], axis=1)) * 180 / np.pi > 2.0:
            #print(i,end=",")
            pass

        if np.max(np.linalg.norm(episode_data["omega"], axis=1)) * 180 / np.pi > 11.0:
            #print(i,end=",")
            pass

        if i == 238:
            #print(f"{i}:\n{episode_data['filter_log']}")
            pass
        
        # if the last attitude error is above 5 degree and the last 50 omega norms are below 0.1 deg/s each
        omega_settled = True
        for omega in episode_data["omega"][-200:]:
            if np.linalg.norm(omega)*180/np.pi > 2.0:
                omega_settled = False
        if 2 * np.arccos(np.abs(episode_data["quaternion"][-1,0])) * 180/np.pi < 5.0 and omega_settled:
            #print(i,end=",")
            settled_count += 1
            settled_final_err.append(2 * np.arccos(np.abs(episode_data["quaternion"][-1,0])) * 180/np.pi)
            pass

        if not omega_settled:
            #print(i,end=",")
            pass
        
        if reward_min is None:
            reward_min = episode_data["cumulative_rewards"][-1]
            reward_min_idx = i
        if episode_data["cumulative_rewards"][-1] < reward_min:
            #print(i,end=",")
            reward_min = episode_data["cumulative_rewards"][-1]
            reward_min_idx = i
            pass
    print()
    # print(len(data))
    # print("Settled episodes:", settled_count)
    # print("Mean final error of settled episodes:", np.mean(settled_final_err))
    # print("--------------------------------------------------")
    #print(reward_min_idx, reward_min)
    return data

def plot_for_report(simulation_data: dict, time_end=300):
    """Plot the data and arrange it for report format."""
    
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
    normal_vector_koz = simulation_data["normal_vector_koz"]
    half_angle_koz = simulation_data["half_angle_koz"]
    margin_angles_koz = simulation_data["margin_angles_koz"]
    min_margin_koz = simulation_data["min_margin_koz"]
    cnt_Koz_violations = simulation_data["cnt_Koz_violations"]

    print("Minimum margin KOZ:", min_margin_koz*180/np.pi, "degrees")
    print("Count KOZ violations:", cnt_Koz_violations)
    print("Half angle KOZ:", half_angle_koz*180/np.pi, "degrees")
    

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
    
    # figure for the trajectory
    fig1 = plt.figure(figsize=(8, 8))
    
    # 3D Rotation Axis Trajectory (This is the key trajectory for phi angle!)
    ax1 = fig1.add_subplot(111, projection="3d")
    
    # Adjust subplot to fill more of the figure space
    fig1.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Plot trajectory on unit sphere (rotation axes are unit vectors)
    ax1.plot(body_axis_arr[:, 0], body_axis_arr[:, 1], body_axis_arr[:, 2], "b-", alpha=0.7, linewidth=3, label="Boresight axis trajectory")
    ax1.scatter(body_axis_arr[0, 0], body_axis_arr[0, 1], body_axis_arr[0, 2], color="green", s=50, label="Start")
    ax1.scatter(body_axis_arr[-1, 0], body_axis_arr[-1, 1], body_axis_arr[-1, 2], color="red", s=50, label="End")
    # Plot target last with higher zorder to ensure it's always on top
    ax1.scatter(1, 0, 0, color="gold", s=200, marker="*", label="Target", zorder=1000, edgecolors='black', linewidths=1)

    def _generate_keep_out_zone_circle():
        # Create circle points for the keep out zone
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_points = []
        for angle in theta:
            # Generate points on the circle in the plane perpendicular to the normal vector
            v = np.array([np.cos(angle), np.sin(angle), 0])
            # Rotate v to be perpendicular to koz_normal
            if np.allclose(normal_vector_koz, [0, 0, 1]):
                rot_axis = np.array([1, 0, 0])
            else:
                rot_axis = np.cross([0, 0, 1], normal_vector_koz)
                rot_axis /= np.linalg.norm(rot_axis)
            angle_to_rotate = np.arccos(np.dot(normal_vector_koz, [0, 0, 1]))
            # Rodrigues' rotation formula
            v_rotated = (v * np.cos(angle_to_rotate) +
                        np.cross(rot_axis, v) * np.sin(angle_to_rotate) +
                        rot_axis * np.dot(rot_axis, v) * (1 - np.cos(angle_to_rotate)))
            # Scale to the radius of the keep out zone circle
            radius = np.sin(half_angle_koz)
            circle_point = normal_vector_koz * np.cos(half_angle_koz) + v_rotated * radius
            circle_points.append(circle_point)

        return circle_points

    # Plot keep out zone as a ring on the unit sphere
    if normal_vector_koz is not None and half_angle_koz is not None:
        circle_points = _generate_keep_out_zone_circle()
        circle_points = np.array(circle_points)
        ax1.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], "orange", linewidth=2, label="Keep-out zone")
    
    # Draw unit sphere wireframe
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_wireframe(sphere_x, sphere_y, sphere_z, alpha=0.1, color="gray")
    
    # Remove cartesian grid (x, y, z panes and axes)
    ax1.grid(False)
    ax1.set_axis_off()
  
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 0.2), fontsize=6)
    
    # cut time and data
    times = times[:int(time_end/dt)]
    q_0 = q_0[:int(time_end/dt)]
    q_1 = q_1[:int(time_end/dt)]
    q_2 = q_2[:int(time_end/dt)]
    q_3 = q_3[:int(time_end/dt)]
    omega_x = omega_x[:int(time_end/dt)]
    omega_y = omega_y[:int(time_end/dt)]
    omega_z = omega_z[:int(time_end/dt)]
    torques_array = torques_array[:int(time_end/dt)]
    margin_angles_koz = margin_angles_koz[:int(time_end/dt)]

    # figure for attitude, angular velocity, torque and KOZ margin
    fig2 = plt.figure(figsize=(6, 6))
    
    # Plot quaternion
    ax4 = fig2.add_subplot(411)
    ax4.plot(times, q_0, label="$q_0$")
    ax4.plot(times, q_1, label="$q_1$")
    ax4.plot(times, q_2, label="$q_2$")
    ax4.plot(times, q_3, label="$q_3$")
    ax4.set_ylabel("q")
    ax4.legend(loc="upper right")
    ax4.grid()

    # Plot angular velocity
    ax5 = fig2.add_subplot(412)
    ax5.plot(times, omega_x * (180 / np.pi), label="$\\omega_x$")
    ax5.plot(times, omega_y * (180 / np.pi), label="$\\omega_y$")
    ax5.plot(times, omega_z * (180 / np.pi), label="$\\omega_z$")
    ax5.set_ylabel("$\\omega$ [deg/s]")
    ax5.legend(loc="upper right")
    ax5.grid()
    # Plot torque input
    ax6 = fig2.add_subplot(413)
    ax6.plot(times, torques_array[:, 0], label="$\\tau_1$")
    ax6.plot(times, torques_array[:, 1], label="$\\tau_2$")
    ax6.plot(times, torques_array[:, 2], label="$\\tau_3$")
    ax6.set_ylabel("$\\tau$ [Nm]")
    #ax6.set_xlabel("Time [s]")
    ax6.legend(loc="upper right")
    ax6.grid()

    # Plot keep out zone margin angle
    ax7 = fig2.add_subplot(414)
    ax7.plot(times, margin_angles_koz, label="$\\theta_{margin}$")
    ax7.set_xlabel("Time [s]")
    ax7.set_ylabel("$\\theta_{margin}$ [deg]")
    ax7.legend(loc="upper right")
    ax7.grid()
    
    plt.tight_layout()
    plt.show()
    
    return

def calc_std_deviation(data: list):
    """
    Calculate some metrics across the episodes.
    Inputs:
        data: List of episode data dictionaries.
    """
  
    episode_rewards = []
    koz_margin_violated = []
    koz_margin_not_violated = []

    settled_count = 0
    settled_final_err = []
    settling_time = []
    control_effort = []

    alt_settled_count = 0
    alt_settled_final_err = []
    alt_settling_time = []
    alt_control_effort = []
    

    for i, episode_data in enumerate(data):
        episode_rewards.append(episode_data["cumulative_rewards"][-1])

        # considered settled, if the final attitude error is below 5 degree and the angular velocity (norm) is below 2 deg/s for the last 200 time steps
        # check if velocity settled
        omega_settled = True
        for omega in episode_data["omega"][-200:]:
            if np.linalg.norm(omega)*180/np.pi > 2.0:
                omega_settled = False

        # if attitude and velocity is settled, include pointing accuracy, settling time and control effort
        if 2 * np.arccos(np.abs(episode_data["quaternion"][-1,0])) * 180/np.pi < 5.0 and omega_settled:
            #print(i,end=",")
            settled_count += 1
            settled_final_err.append(2 * np.arccos(np.abs(episode_data["quaternion"][-1,0])) * 180/np.pi)
            #settling_time.append(episode_data["time"][-1])
            control_effort.append(np.sum(np.linalg.norm(episode_data["torques"], axis=1)**2))
            pass

        # alternative settling definition: settled if attitude error stays below target accuracy (0.25 deg) after some time (settling time).
        tmp_settling_time = 0
        tmp_settled = False

        for idx, quat in enumerate(episode_data["quaternion"]):
            err_angle = 2 * np.arccos(np.abs(quat[0])) * 180/np.pi
            if err_angle <= 0.25:
                if not tmp_settled:
                    tmp_settling_time = idx * dt
                tmp_settled = True
            else:
                tmp_settled = False

        # If settled, include these metrics
        if tmp_settled:
            alt_settled_count += 1
            alt_settling_time.append(tmp_settling_time)
            alt_settled_final_err.append(2 * np.arccos(np.abs(episode_data["quaternion"][-1,0])) * 180/np.pi)
            alt_control_effort.append(np.sum(np.linalg.norm(episode_data["torques"], axis=1)**2))

        if episode_data["min_margin_koz"]*180/np.pi < 0.0:
            koz_margin_violated.append(episode_data["min_margin_koz"]*180/np.pi)
        else:
            koz_margin_not_violated.append(episode_data["min_margin_koz"]*180/np.pi)
       
    print()
    print("Episodes: ", len(data))
    print()
    print(f"KOZ margin violated: {len(koz_margin_violated)} episodes")
    print(f"KOZ margin not violated: {len(koz_margin_not_violated)} episodes")
    if len(koz_margin_violated) > 0:
        print(f"Mean KOZ margin violated: {np.mean(koz_margin_violated)} +- {np.std(koz_margin_violated)} degrees")
    if len(koz_margin_not_violated) > 0:
        print(f"Mean KOZ margin not violated: {np.mean(koz_margin_not_violated)} +- {np.std(koz_margin_not_violated)} degrees")
    print()
    print("Settled episodes:", settled_count)
    print(f"Mean final error of settled episodes: {np.mean(settled_final_err)} +- {np.std(settled_final_err)} degrees")
    #print(f"Mean settling time: {np.mean(settling_time)} +- {np.std(settling_time)} seconds")
    print(f"Mean control effort: {np.mean(control_effort)} +- {np.std(control_effort)} N^2m^2*s")
    print()
    print(f"Alt settled episodes: {alt_settled_count}")
    print(f"Alt Mean final error of settled episodes: {np.mean(alt_settled_final_err)} +- {np.std(alt_settled_final_err)} degrees")
    print(f"Alt Mean settling time: {np.mean(alt_settling_time)} +- {np.std(alt_settling_time)} seconds")
    print(f"Alt Mean control effort: {np.mean(alt_control_effort)} +- {np.std(alt_control_effort)} N^2m^2*s")
    print(f"Rewards: {np.mean(episode_rewards)} +- {np.std(episode_rewards)}")
    print("--------------------------------------------------")
    

### MAIN ###
if __name__ == "__main__":
    MODEL_NAME = "phase1_best1_ph2_sfty2_11100000"
    model = load_agent(MODEL_NAME)
    MAX_STEPS = 3000

    # Set initial state for evaluation environment
    INITIAL_STATE = [80.0, 180.0, 0.00, 0.01, MAX_STEPS, 15.0, 30.0]  # [min_initial_angle, max_initial_angle, min_initial_angular_velocity, max_initial_angular_velocity]

    USE_SAFETY_FILTER = 2  # 0: no filter, 1: filter applied, 2: train with filter

    eval_env = create_evaluation_env(INITIAL_STATE, USE_SAFETY_FILTER)

    """ Uncomment simulate_agent() and plot_actual_attitude() below to run 1 simulation and plot the results. """
    #print_rewards(model, eval_env, n_eval_episodes=1)
    #simulation_data = simulate_agent(model, eval_env, MAX_STEPS)
    #plot_actual_attitude(simulation_data)
    #plot_for_report(simulation_data, time_end=300)

    """ Uncomment the lines below if you have saved evaluation data (from evaluate_agent()) to load all the episodes.
        loaded contains ALL episodes, therefore in loaded[] should be the index of the episode you want to plot.
    """
    #loaded = load_evaluation_data("evaluation_phase1_best1_backup_noFilter_initRange_10k.npz")
    #loaded = load_evaluation_data("evaluation_phase1_best1_backup_filter_initRange_10k_fixed.npz")

    #loaded = load_evaluation_data("evaluation_phase1_best1_3_9.6M_noFilter_initRange_10k.npz")
    #loaded = load_evaluation_data("evaluation_phase1_best1_3_9.6M_filter_initRange_10k_fixed_noViolation.npz")

    #loaded = load_evaluation_data("evaluation_phase1_best1_ph2_sfty2_noFilter_initRange_10k.npz")
    loaded = load_evaluation_data("evaluation_phase1_best1_ph2_sfty2_filter_initRange_10k_fixed_11M.npz")
    
    #calc_std_deviation(loaded)
    #plot_actual_attitude(loaded[2654])
    plot_for_report(loaded[4149],time_end=300)

    """ Uncomment evaluate_agent() below to simulate the agent over multiple episodes and save the data at the end. """
    t_start = time.time()
    # Run evaluation with k parallel workers and n episodes
    #evaluate_agent(MODEL_NAME, INITIAL_STATE, USE_SAFETY_FILTER, MAX_STEPS, episodes=10000, num_workers=4)
    t_end = time.time()
    print()
    print(f"Evaluation time: {t_end - t_start:.2f} seconds")

