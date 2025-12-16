import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from environment import SatDynEnv, scale_torque

def create_simulation_env(initial_state):
    """
    Create the simulation environment.
    """
    sim_env = SatDynEnv(render_mode="rgb_array", initial_state=initial_state)
    return sim_env


def action_schedule(t):
    """
    Yields an action based on the current time t.
    Actions are between -1 and 1 and correspond to -0.0007 Nm and +0.0007 NM
    """
    action = np.zeros(4)

    if 2.0 <= t < 10.0:
        action[0] = 1

    return action


def start_simulation(env: SatDynEnv):
    """
    Start the simulation in the given environment.
    """
    observation = env.reset()
    times = np.linspace(0, env.max_steps/10, env.max_steps)  # Assuming dt=0.1s
    states= []
    torques = []

    done = False
    while not done:
        action = action_schedule(env.steps * env.dt)
        observation, reward, done, truncated, info = env.step(action)

        states.append(observation)
        torques.append(scale_torque*action)
    env.close()

    states_array =  np.array(states)
    torques_array = np.array(torques)

    # store the norm of quaternions
    norm_q = np.linalg.norm(states_array[:, :4], axis=1)

    simulation_data = {
        "quaternion": states_array[:, :4],
        "quaternion_norm": norm_q,
        "torques": torques_array,
        "omega": states_array[:, 4:7],
        "times": times
        }
    
    return simulation_data


def plot_actual_attitude(simulation_data: dict):
    """Plot the ACTUAL satellite attitude trajectory based on rotation axis and angle phi"""

    matplotlib.use("TkAgg")

    # Parse quaternion and angular velocity components
    q_0, q_1, q_2, q_3 = simulation_data["quaternion"][:, 0], simulation_data["quaternion"][:, 1], simulation_data["quaternion"][:, 2], simulation_data["quaternion"][:, 3]
    omega_x, omega_y, omega_z = simulation_data["omega"][:, 0], simulation_data["omega"][:, 1], simulation_data["omega"][:, 2]
    norm_q = simulation_data["quaternion_norm"]
    torques_array = simulation_data["torques"]
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
    
    fig = plt.figure(figsize=(18, 12))
    
    # 3D Rotation Axis Trajectory (This is the key trajectory for phi angle!)
    ax1 = fig.add_subplot(231, projection="3d")
    
    # Plot trajectory on unit sphere (rotation axes are unit vectors)
    ax1.plot(body_axis_arr[:, 0], body_axis_arr[:, 1], body_axis_arr[:, 2], "b-", alpha=0.7, linewidth=3, label="Boresight Axis Trajectory")
    ax1.scatter(body_axis_arr[0, 0], body_axis_arr[0, 1], body_axis_arr[0, 2], color="green", s=100, label="Start")
    ax1.scatter(body_axis_arr[-1, 0], body_axis_arr[-1, 1], body_axis_arr[-1, 2], color="red", s=100, label="End")
    ax1.scatter(1, 0, 0, color="gold", s=150, marker="*", label="Target")
    
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
    ax2 = fig.add_subplot(232)
    ax2.plot(times[:len(rotation_angles_deg)], rotation_angles_deg, "purple", linewidth=3, label="Angle $\\phi$")
    ax2.axhline(y=2.0, color="r", linestyle="--", linewidth=2, label="Target (2.0°)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Rotation Angle $\\phi$ (°)")
    ax2.set_title("Rotation Angle $\\phi$ vs Time\n(Used in reward function)")
    ax2.grid(True)
    ax2.legend()
    ax2.set_yscale("log")
    
    # Plot quaternion
    ax4 = fig.add_subplot(234)
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
    ax5 = fig.add_subplot(235)
    ax5.plot(times, omega_x * (180 / np.pi), label="$\\omega_x$")
    ax5.plot(times, omega_y * (180 / np.pi), label="$\\omega_y$")
    ax5.plot(times, omega_z * (180 / np.pi), label="$\\omega_z$")
    ax5.set_title("Angular velocity")
    ax5.set_ylabel("$\\omega$ (deg/s)")
    ax5.legend()
    ax5.grid()
    # Plot torque input
    ax6 = fig.add_subplot(236)
    ax6.plot(times, torques_array[:, 0], label="$\\tau_1$")
    ax6.plot(times, torques_array[:, 1], label="$\\tau_2$")
    ax6.plot(times, torques_array[:, 2], label="$\\tau_3$")
    ax6.plot(times, torques_array[:, 3], label="$\\tau_4$")
    ax6.set_title("Control torques")
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("$\\tau$ (Nm)")
    ax6.legend()
    ax6.grid()
    
    plt.tight_layout()
    plt.show()
    
    return

if __name__ == "__main__":
    # Set the torques in action_schedule()
    initial_state = [0.0, 0.0, 0.0, 0.0, 300] # [min_initial_angle, max_initial_angle, min_initial_angular_velocity, max_initial_angular_velocity, max_steps]
    env = create_simulation_env(initial_state)
    simulation_data = start_simulation(env)
    plot_actual_attitude(simulation_data)