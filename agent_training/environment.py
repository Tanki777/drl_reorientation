import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend for frame rendering

import math

from numba import njit


kp = 50
kd = 500

scale_torque = 0.0007


@njit
def sign_fun(x):
    if x >= 0:
        sign = 1
    else:
        sign = -1
    return sign

@njit
def normalize_quaternion(q):
    #norm = np.linalg.norm(q)
    norm = np.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)   # using custom calculation of norm in order to use numba
    if norm > 0:  # Avoid division by zero
        return q / norm
    return q  # Return unchanged if norm is zero

@njit
def normalize_vector(v):
    #norm = np.linalg.norm(v)
    norm = np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)   # using custom calculation of norm in order to use numba
    if norm > 0:  # Avoid division by zero
        return v / norm
    return v  # Return unchanged if norm is zero

@njit
def torque_function(state, kp, kd):
    # The desired quaternion is the identity quaternion
    q0 = state[0]
    q_vec = state[1:4]
    omega_vec = state[4:7]
        
    torque = -kp * sign_fun(q0) * q_vec - kd * omega_vec
    return torque

@njit
def quaternion_multiply(q1,q2):
    """Multiply two quaternions q1 and q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ], dtype=np.float32)


def rotate_vector_by_quaternion(v, q):
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x*x + y*y)]
    ])

    return R.dot(v)

@njit
def sat_ode(state, inertia_total, inertia_wheels, wheels_position_matrix, torque):

    state = state.astype(np.float32)
    inertia_total = inertia_total.astype(np.float32)
    inertia_wheels = inertia_wheels.astype(np.float32)
    wheels_position_matrix = wheels_position_matrix.astype(np.float32)
    torque = torque.astype(np.float32)

    q_quat = state[:4]  # Quaternion
    omega = state[4:7]
    wheel_velocities = state[15:19]

    # from paper 2023 equation 2a
    omega_cross = np.array([[0, -omega[0], -omega[1], -omega[2]],
                            [omega[0], 0, omega[2], -omega[1]],
                            [omega[1], -omega[2], 0, omega[0]],
                            [omega[2], omega[1], -omega[0], 0]],
                           np.float32)
    
    # Z^-1 matrix from paper 2023 equation 2b
    inertia_combined = np.array([[inertia_total, wheels_position_matrix @ inertia_wheels],
	                  [inertia_wheels @ wheels_position_matrix.transpose(), inertia_wheels]], np.float32)

    # Z matrix from paper 2023 equation 2b
    inertia_combined_inv = np.linalg.inv(inertia_combined)

	# from paper 2023 equation 2a
    q_dot = 0.5 * omega_cross @ q_quat.reshape(-1, 1)

    # vector, which is multiplied to Z matrix from paper 2023 equation 2b
    top = np.cross(-omega, (inertia_total @ omega + wheels_position_matrix @ inertia_wheels @ wheel_velocities))# + xi   # (3,) size
    bottom = torque                                                     # (4,)
    vector_2b = np.vstack([top.reshape(3,1), bottom.reshape(3,1)])         # (7,1) column

    dynamics = inertia_combined_inv @ vector_2b 						 # (7,1) column of combined results
    omega_dot = dynamics[0:3].reshape(3,)   # first 3 entries
    wheel_velocities_dot = dynamics[3:7].reshape(4,)   # last 4 entries

    # flatten() turns into 1D array, concatenate() joins them all together
    return np.concatenate((q_dot.flatten(), omega_dot.flatten(), wheel_velocities_dot.flatten())) 


@njit
def reward_function(state):
    q0_current = state[0]
    q0_prev = state[7]
    torque_x = state[11]
    torque_y = state[12]
    torque_z = state[13]
    torque_x_prev = state[14]
    torque_y_prev = state[15]
    torque_z_prev = state[16]
    
    err_phi_current = 2 * math.acos(q0_current)   # in [rad]
    err_phi_prev = 2 * math.acos(q0_prev)   # in [rad]

    if err_phi_current <= err_phi_prev:
        reward0 = math.exp(-err_phi_current/(0.14 * 2 * np.pi)) - 0.05*(math.sqrt(torque_x**2 + torque_y**2 + torque_z**2)/math.sqrt(12)) - 0.005*math.sqrt((torque_x - torque_x_prev)**2 + (torque_y - torque_y_prev)**2 + (torque_z - torque_z_prev)**2)
    else:
        reward0 = math.exp(-err_phi_current/(0.14 * 2 * np.pi)) - 0.05*(math.sqrt(torque_x**2 + torque_y**2 + torque_z**2)/math.sqrt(12)) - 0.005*math.sqrt((torque_x - torque_x_prev)**2 + (torque_y - torque_y_prev)**2 + (torque_z - torque_z_prev)**2) - 1

    if err_phi_current <= 0.25 * np.pi / 180:       # required attitude accuracy is satisfied
        return reward0 + 9
    else:
        return reward0


class SatDynEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode=None, initial_state=None):
        super(SatDynEnv).__init__()

        # Define action space as [torque_1, torque_2, torque_3, torque_4]  (torques of 4 reaction wheels)
        # Value range [-scale_torque, scale_torque] for each component
        """ TO DO: to normalize the action space"""
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        # Define observation space: [q, omega,  q0_prev] = [q_0, q_1, q_2, q_3, omega_1, omega_2, omega_3, q0_prev, omega_1_prev, omega_2_prev, omega_3_prev, torque_x, torque_y, torque_z, torque_x_prev, torque_y_prev, torque_z_prev, ]
        # previous q0 is augmented in the state vector since it will be used in the reward function.
        self.observation_space = spaces.Box(low= np.array([-1, -1, -1, -1, -300, -300, -300, -1, -300, -300, -300, -2, -2, -2, -2, -2, -2]),
                                            high= np.array([1, 1, 1, 1, 300, 300, 300, 1, 300, 300, 300, 2, 2, 2, 2, 2, 2]),
                                            dtype= np.float32)

        # Initial state
        self.render_mode = render_mode
        
        # Randomization parameters for robust training
        if initial_state is None:
            self.min_initial_angle = 0.0  # degrees - minimum initial attitude error
            self.max_initial_angle = 90.0  # degrees - maximum initial attitude error
            self.min_initial_angular_velocity = 0.0  # deg/s - minimum initial tumbling rate
            self.max_initial_angular_velocity = 0.1  # deg/s - maximum initial tumbling rate
        else:
            self.min_initial_angle = initial_state[0]
            self.max_initial_angle = initial_state[1]
            self.min_initial_angular_velocity = initial_state[2]
            self.max_initial_angular_velocity = initial_state[3]
        
        # Custom metrics tracking for TensorBoard
        self.initial_error_angle = 0.0
        self.initial_angular_velocity_mag = 0.0
        self.episode_torques = []
        self.episode_torques_prev = []
        self.settled = False
        self.settling_time = -1  # -1 means not settled
        self.settling_threshold_deg = 2.0  # degrees for considering "settled"
        self.settling_velocity_threshold = 0.01  # rad/s for angular velocity
        
        # Set initial state (will be randomized in reset())
        self.reset()

        self.inertia_body = np.array([[1.672, 0.0, 0.0],
                                [0.0, 0.1259, 0.0],
                                [0.0, 0.0, 0.06121]],
                                np.float32)
        
        self.inertia_wheels = 0.00001722

        self.wheels_positions = np.array([[0.0, 0.0, 0.8165, -0.8165],
                                        [0.0, -0.9428, 0.4714, 0.4714],
                                        [-1.0, 0.3333, 0.3333, 0.3333]],
                                np.float32)
        
        # see 3 lines above dynamics equations red box, in 2023 paper
        # the sum_i (a_i)(a_i)T is an outer product, equal to A @ A.T
        self.inertia_total = self.inertia_body + self.inertia_wheels * (self.wheels_positions @ self.wheels_positions.transpose()) 

        # Define time step, step duration, and maximum steps
        self.dt = 0.1
        self.max_steps = 1000
        self.steps = 0
        self.render_mode = render_mode

        self.x_axis = np.array([1, 0, 0]) # For frame rendering

    def _generate_random_quaternion(self, min_angle_deg, max_angle_deg):
        """Generate a random quaternion representing rotation within max_angle_deg from identity"""
        if max_angle_deg == 0:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Random rotation angle between min_angle_deg and max_angle_deg
        angle = np.random.uniform(min_angle_deg * np.pi / 180, max_angle_deg * np.pi / 180)
        
        # Random rotation axis (uniformly distributed on unit sphere)
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        
        # Convert axis-angle to quaternion
        q0 = np.cos(angle / 2)
        q_vec = np.sin(angle / 2) * axis
        
        quaternion = np.array([q0, q_vec[0], q_vec[1], q_vec[2]], dtype=np.float32)
        return normalize_quaternion(quaternion)  # Normalize

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random initial attitude error (0° to max_initial_angle)
        q_array_initial = self._generate_random_quaternion(self.min_initial_angle, self.max_initial_angle)
        
        # Generate random initial angular velocities
        omega_min_rad = self.min_initial_angular_velocity * np.pi / 180  # Convert to rad/s
        omega_max_rad = self.max_initial_angular_velocity * np.pi / 180  # Convert to rad/s
        
        # Generate random magnitudes between min and max
        omega_magnitude = np.random.uniform(omega_min_rad, omega_max_rad)
        
        # Generate random direction (uniformly distributed on unit sphere)
        omega_direction = np.random.randn(3)
        omega_direction = omega_direction / np.linalg.norm(omega_direction)
        
        # Scale direction by magnitude
        omega_initial = (omega_magnitude * omega_direction).astype(np.float32)
        
        q0_prev = q_array_initial[0]
        omega_prev = omega_initial
        state_ = np.concatenate((q_array_initial, omega_initial))
        self.state = np.concatenate((state_, [q0_prev, *omega_prev, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

        self.steps = 0
        
        # Initialize custom metrics for this episode
        self.initial_error_angle = 2 * math.acos(abs(q_array_initial[0])) * 180 / np.pi  # degrees
        self.initial_angular_velocity_mag = np.linalg.norm(omega_initial) * 180 / np.pi  # deg/s
        self.episode_torques = []
        self.episode_torques_prev = []
        self.settled = False
        self.settling_time = -1
        
        return self.state, {}

    def step(self, action):
        inertia_inv = np.linalg.inv(self.inertia)

        q0_prev = self.state[0]     # store current q0 before integration
        omega_prev = self.state[4:7]  # store current omega before integration
        torque_prev = self.state[11:14]  # store current torque before integration

        """ integrating using 4th-order RK method """
        f1 = self.dt * sat_ode(self.state[:7], self.inertia_total, self.inertia_wheels, self.wheels_positions, action * scale_torque)
        f2 = self.dt * sat_ode(self.state[:7] + 0.5 * f1, self.inertia_total, self.inertia_wheels, self.wheels_positions, action * scale_torque)
        f3 = self.dt * sat_ode(self.state[:7] + 0.5 * f2, self.inertia_total, self.inertia_wheels, self.wheels_positions, action * scale_torque)
        f4 = self.dt * sat_ode(self.state[:7] + f3, self.inertia_total, self.inertia_wheels, self.wheels_positions, action * scale_torque)

        self.state[:7] = self.state[:7] + (f1 + 2 * f2 + 2 * f3 + f4)/6

        # Normalize quaternion after integration
        self.state[:4] = normalize_quaternion(self.state[:4])

        self.state[7] = q0_prev
        self.state[8:11] = omega_prev
        self.state[14:17] = torque_prev
        applied_torque = action * scale_torque
        self.state[11:14] = applied_torque
        

        # Calculate reward
        reward = reward_function(self.state)
        
        # Track custom metrics
        
        self.episode_torques.append(np.linalg.norm(applied_torque))
        self.episode_torques_prev.append(np.linalg.norm(torque_prev))

        # Explicitly cast the state to float32 before returning
        obs = self.state.astype(np.float32)
        
        # Check settling condition
        current_error_deg = 2 * math.acos(abs(self.state[0])) * 180 / np.pi
        current_omega_mag = np.linalg.norm(self.state[4:7])
        
        if (not self.settled and 
            current_error_deg < self.settling_threshold_deg and 
            current_omega_mag < self.settling_velocity_threshold):
            self.settled = True
            self.settling_time = self.steps * self.dt  # settling time in seconds

        # Check if the maximum number of steps is reached
        self.steps += 1
        done = self.steps >= self.max_steps
        truncated = False
        
        # Prepare info dict with custom metrics
        info = {}
        
        # Add episode-end metrics when episode is done
        if done:
            final_error_angle = current_error_deg
            avg_torque = np.mean(self.episode_torques) if self.episode_torques else 0.0
            max_torque = np.max(self.episode_torques) if self.episode_torques else 0.0
            max_torque_prev = np.max(self.episode_torques_prev) if self.episode_torques else 0.0
            
            info.update({
                "custom_metrics/initial_error_angle": self.initial_error_angle,
                "custom_metrics/initial_angular_velocity": self.initial_angular_velocity_mag,
                "custom_metrics/final_error_angle": final_error_angle,
                "custom_metrics/settling_time": self.settling_time,
                "custom_metrics/avg_torque": avg_torque,
                "custom_metrics/max_torque": max_torque,
                "custom_metrics/max_torque_prev": max_torque_prev,
                "custom_metrics/settled": float(self.settled),
            })

        return obs, reward, done, truncated, info

    def render(self):
        attitude = self.state[:4]
        omega = self.state[4:7]
        torque = torque_function(self.state, kp, kd)

        if self.render_mode == "human":
            print(f"Step: {self.steps}, Attitude: {attitude}, Omega: {omega}, Torque: {torque}")
            return

        if self.render_mode == "rgb_array":
            q = self.state[:4]

            # Rotate the satellite body axis (x-axis) by the quaternion
            body_axis = rotate_vector_by_quaternion(self.x_axis, q)

            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=30, azim=135)

            # Draw world x-axis (target axis)
            ax.quiver(0, 0, 0, 1, 0, 0, color="red")

            # Draw the satellite body axis
            ax.quiver(0, 0, 0, body_axis[0], body_axis[1], body_axis[2], color="black", linewidth=3)

            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_box_aspect([1, 1, 1])

            # Convert the figure to an RGB array
            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
            plt.close(fig)

            return frame

    def close(self):
        pass