import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend for frame rendering

from stable_baselines3.common.callbacks import BaseCallback

import math

import time

import os

from numba import njit


kp = 50
kd = 500

scale_torque = 2

class CustomCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1, total_timesteps_offset=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.base_save_path = save_path
        self.cumulative_reward = 0     # Initialize cumulative reward
        self.total_timesteps_offset = total_timesteps_offset  # Track total across sessions

    def _on_step(self) -> bool:
        # Access the immediate reward from locals
        immediate_reward = self.locals.get('rewards')

        # Update cumulated reward
        if immediate_reward is not None:
            self.cumulative_reward += immediate_reward

        if self.n_calls % self.check_freq == 0:
            # Create a unique model filename using timestamp
            timestamp = int(time.time())

            # Calculate true total timesteps across all sessions
            true_total_timesteps = self.num_timesteps + self.total_timesteps_offset
            unique_model_path = f"{self.base_save_path}_{true_total_timesteps}_{timestamp}"

            # Save the model
            self.model.save(unique_model_path)

            # Save relevant information with true total timesteps
            with open("training_logs_numba.txt", "a") as log_file:
                log_file.write(f"Step: {true_total_timesteps}, Cumulative Reward: {self.cumulative_reward}\n")

            self.cumulative_reward = 0      # Reset cumulative reward after logging
        return True


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
def sat_ode(state, inertia, inertia_inv, torque):

    state = state.astype(np.float32)
    inertia = inertia.astype(np.float32)
    inertia_inv = inertia_inv.astype(np.float32)
    torque = torque.astype(np.float32)

    q_quat = state[:4]  # Quaternion
    omega = state[4:7]

    omega_cross = np.array([[0, -omega[2], omega[1]],
                            [omega[2], 0, -omega[0]],
                            [-omega[1], omega[0], 0]],
                           np.float32)

    omega_dot = inertia_inv @ (-omega_cross @ inertia @ omega.reshape(-1, 1) + torque.reshape(-1, 1))

    omega_quat = np.array([0.0, omega[0], omega[1], omega[2]], dtype=np.float32)  # Omega as vector quaternion
    q_dot = 0.5 * quaternion_multiply(q_quat, omega_quat)

    return np.concatenate((q_dot, omega_dot.flatten()))

@njit
def reward_function(state):
    q0_current = state[0]
    omega_x = state[4]
    omega_y = state[5]
    omega_z = state[6]
    
    # Attitude error (rotation angle from identity quaternion)
    err_phi = 2 * math.acos(abs(q0_current))  # [rad]
    err_phi_deg = err_phi * 180 / np.pi  # [degrees]
    
    # Angular velocity magnitude
    omega_mag = math.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
    
    # Coarse attitude reward
    attitude_reward = -err_phi_deg / 10.0  # -18 to 0 for 0° to 180° error
    
    # Fine attitude reward
    if err_phi_deg < 10.0:  # Within 10 degrees
        fine_reward = 10 * math.exp(-err_phi_deg / 2.0)  # 10 to ~0
    else:
        fine_reward = 0
    
    # Angular velocity penalty for settling
    velocity_penalty = -0.1 * omega_mag
    
    # Success bonus
    if err_phi_deg < 2.0:  # 2° accuracy
        success_bonus = 50
    elif err_phi_deg < 5.0:  # 5° accuracy  
        success_bonus = 20
    else:
        success_bonus = 0
        
    # High accuracy bonus
    if err_phi_deg < 0.5:  # 0.5°
        accuracy_bonus = 100
    else:
        accuracy_bonus = 0
    
    total_reward = attitude_reward + fine_reward + velocity_penalty + success_bonus + accuracy_bonus
    
    return total_reward

class SatDynEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode=None):
        super(SatDynEnv).__init__()

        # Define action space as [torque_x, torque_y, torque_z]  (torques about three body axes)
        # Value range [-2, 2] for each component
        """ TO DO: to normalize the action space"""
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # Define observation space: [q, omega, q0_prev] = [q_0, q_1, q_2, q_3, omega_1, omega_2, omega_3, q0_prev]
        # previous q0 is augmented in the state vector since it will be used in the reward function.
        self.observation_space = spaces.Box(low= np.array([-1, -1, -1, -1, -300, -300, -300, -1]),
                                            high= np.array([1, 1, 1, 1, 300, 300, 300, 1]),
                                            dtype= np.float32)

        # Initial state
        self.render_mode = render_mode
        
        # Randomization parameters for robust training
        self.max_initial_angle = 90.0  # degrees - maximum initial attitude error
        self.max_initial_angular_velocity = 0.1  # deg/s - maximum initial tumbling rate
        
        # Set initial state (will be randomized in reset())
        self.reset()

        self.inertia = np.array([[60, 5, 1],
                                [5, 50, 2],
                                [1, 2, 70]],
                                np.float32)
        """TO DO: to add random noises on the nominal inertia matrix"""

        # Define time step, step duration, and maximum steps
        self.dt = 0.1
        self.max_steps = 1000
        self.steps = 0
        self.render_mode = render_mode

        self.x_axis = np.array([1, 0, 0]) # For frame rendering

    def _generate_random_quaternion(self, max_angle_deg):
        """Generate a random quaternion representing rotation within max_angle_deg from identity"""
        if max_angle_deg == 0:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Random rotation angle between 0 and max_angle_deg
        angle = np.random.uniform(0, max_angle_deg * np.pi / 180)
        
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
        q_array_initial = self._generate_random_quaternion(self.max_initial_angle)
        
        # Generate random initial angular velocities
        omega_max_rad = self.max_initial_angular_velocity * np.pi / 180  # Convert to rad/s
        omega_initial = np.random.uniform(-omega_max_rad, omega_max_rad, 3).astype(np.float32)
        
        q0_prev = q_array_initial[0]
        state_ = np.concatenate((q_array_initial, omega_initial))
        self.state = np.concatenate((state_, [q0_prev]))

        self.steps = 0
        return self.state, {}

    def step(self, action):
        inertia_inv = np.linalg.inv(self.inertia)

        q0_prev = self.state[0]     # store current q0 before integration

        """ integrating using 4th-order RK method """
        f1 = self.dt * sat_ode(self.state[:7], self.inertia, inertia_inv, action * scale_torque)
        f2 = self.dt * sat_ode(self.state[:7] + 0.5 * f1, self.inertia, inertia_inv, action * scale_torque)
        f3 = self.dt * sat_ode(self.state[:7] + 0.5 * f2, self.inertia, inertia_inv, action * scale_torque)
        f4 = self.dt * sat_ode(self.state[:7] + f3, self.inertia, inertia_inv, action * scale_torque)

        self.state[:7] = self.state[:7] + (f1 + 2 * f2 + 2 * f3 + f4)/6

        # Normalize quaternion after integration
        self.state[:4] = normalize_quaternion(self.state[:4])

        self.state[-1] = q0_prev

        # Explicitly cast the state to float32 before returning
        obs = self.state.astype(np.float32)     #

        # Calculate reward
        reward = reward_function(self.state)

        # Check if the maximum number of steps is reached
        self.steps += 1
        done = self.steps >= self.max_steps

        # terminated = self.steps >= self.max_steps
        truncated = False

        return obs, reward, done, truncated, {}

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