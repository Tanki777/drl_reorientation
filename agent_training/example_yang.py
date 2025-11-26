"""
- adding device setting (cpu or gpu as 'cuda') for training
- adding codes to save relevant information during training
    > 1. using TensorBoard logging
    > 2. creating a custom callback
- Making use of numba for speeding up the model
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

import math

import time

import os

from numba import jit, njit

kp = 50
kd = 500

scale_torque = 5

class CustomCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.base_save_path = save_path
        self.cumulative_reward = 0     # Initialize cumulative reward

    def _on_step(self) -> bool:
        # Access the immediate reward from locals
        immediate_reward = self.locals.get('rewards')

        # Update cumulated reward
        if immediate_reward is not None:
            self.cumulative_reward += immediate_reward

        if self.n_calls % self.check_freq == 0:
            # Create a unique model filename using timestamp
            timestamp = int(time.time())
            unique_model_path = f"{self.base_save_path}_{self.num_timesteps}_{timestamp}"

            # Save the model
            self.model.save(unique_model_path)
            # Save relevant information
            with open("training_logs_numba.txt", "a") as log_file:
                log_file.write(f"Step: {self.num_timesteps}, Cumulative Reward: {self.cumulative_reward}\n")

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

@njit
def sat_ode(state, inertia, inertia_inv, torque):

    state = state.astype(np.float32)
    inertia = inertia.astype(np.float32)
    inertia_inv = inertia_inv.astype(np.float32)
    torque = torque.astype(np.float32)

    #q_quat = Quaternion(state[:4])  # Quaternion
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
    q0_prev = state[-1]

    err_phi_current = 2 * math.acos(q0_current)   # in [rad]
    err_phi_prev = 2 * math.acos(q0_prev)   # in [rad]

    if err_phi_current <= err_phi_prev:
        reward0 = math.exp(-err_phi_current/(0.14 * 2 * np.pi))
    else:
        reward0 = math.exp(-err_phi_current/(0.14 * 2 * np.pi)) - 1

    if err_phi_current <= 0.25 * np.pi / 180:       # required attitude accuracy is satisfied
        return reward0 + 9
    else:
        return reward0

class SatDynEnv(gym.Env):
    def __init__(self):
        super(SatDynEnv).__init__()

        # Define action space as [torque_x, torque_y, torque_z]  (torques about three body axes)
        # Value range [-5, 5] for each component
        """ TO DO: to normalize the action space"""
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # Define observation space: [q, omega, q0_prev] = [q_0, q_1, q_2, q_3, omega_1, omega_2, omega_3, q0_prev]
        # previous q0 is augmented in the state vector since it will be used in the reward function.
        self.observation_space = spaces.Box(low= np.array([-1, -1, -1, -1, -300, -300, -300, -1]),
                                            high= np.array([1, 1, 1, 1, 300, 300, 300, 1]),
                                            dtype= np.float32)

        # Initial state
        """TO DO: to set initial state as random one when implementing RL algorithms"""
        q_array_initial = np.array([0.1531, 0.6853, 0.6953, 0.1531], np.float32)
        omega_initial = np.array([0.5300, 0.5300, 0.053],np.float32) * np.pi / 180    # [rad/s]
        q0_prev = q_array_initial[0]
        state_ = np.concatenate((q_array_initial, omega_initial))
        self.state = np.concatenate((state_, [q0_prev]))

        self.inertia = np.array([[10_000, 0, 0],
                                [0, 9_000, 0],
                                [0, 0, 12_000]],
                                np.float32)
        """TO DO: to add random noises on the nominal inertia matrix"""

        # Define time step, step duration, and maximum steps
        self.dt = 0.1
        self.max_steps = 1000
        self.steps = 0

    def reset(self, seed=None, options=None):
        # Reset attitude and angular rate
        """TO DO: to set initial state as random one when implementing RL algorithms"""
        q_array_initial = np.array([0.1531, 0.6853, 0.6953, 0.1531], np.float32)
        omega_initial = np.array([0.5300, 0.5300, 0.053],np.float32) * np.pi / 180   # [rad/s]
        q0_prev = q_array_initial[0]
        state_ = np.concatenate((q_array_initial, omega_initial))
        self.state = np.concatenate((state_, [q0_prev]))

        self.steps = 0
        return self.state, {}

    def step(self, action):
        inertia_inv = np.linalg.inv(self.inertia)

        q0_prev = self.state[0]     # store current q0 before integration

        """ integrating using 4th-order RK method """
        """
        f1 = self.dt * sat_ode(self.state[:7], self.inertia, inertia_inv, torque_function(self.state[:7], kp, kd))
        f2 = self.dt * sat_ode(self.state[:7] + 0.5 * f1, self.inertia, inertia_inv, torque_function(self.state[:7] + 0.5 * f1, kp, kd))
        f3 = self.dt * sat_ode(self.state[:7] + 0.5 * f2, self.inertia, inertia_inv, torque_function(self.state[:7] + 0.5 * f2, kp, kd))
        f4 = self.dt * sat_ode(self.state[:7] + f3, self.inertia, inertia_inv, torque_function(self.state[:7] + f3, kp, kd))        
        """

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
        """TO DO: to be complete the reward calculation"""
        reward = reward_function(self.state)

        # Check if the maximum number of steps is reached
        self.steps += 1
        done = self.steps >= self.max_steps

        # terminated = self.steps >= self.max_steps
        truncated = False

        return obs, reward, done, truncated, {}

    def render(self, mode="human"):
        #print(f'Step: {self.steps}, Attitude: {self.state[:4]}, Norm: {np.linalg.norm(self.state[:4])}, Torque: {torque_function(self.state, kp, kd)}')
        print(f'Step: {self.steps}, Attitude: {self.state[:4]}, Omega: {self.state[4:7]}, Torque: {torque_function(self.state, kp, kd)}')

    def close(self):
        pass


if __name__ == "__main__":

    print(f"Creating environment.")

    env = SatDynEnv()   # creating environment


    done = False

    # Ensure the directory exists
    if not os.path.exists('models'):
        os.makedirs('models')

    # Setting the path to save the model
    save_path = 'models/ddpg_sat_model_numba'

    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    print(f"Creating the agent.")
    #model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, device='cpu')
    #model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, device='cuda',
    #             tensorboard_log="./ddpg_sat_gpu_temp_numba_tensorboard/")  # creating agent for training using DDPG algorithm; using TensorBoard logging
    custom_callback = CustomCallback(check_freq = 3000, save_path=save_path)    # self-defined callback to save models regularly

    
    # load the existing model for further training    
    model = DDPG.load("ddpg_sat_gpu_2", device='cuda')  # specify the device for gpu
    model.set_env(env)  
    

    print("Start to train the agent >>> ")
    start_time = time.time()
    model.learn(total_timesteps=1_000 * 20, log_interval=10, progress_bar=True, callback=custom_callback)
    end_time = time.time()
    print(f"Time for training: {end_time - start_time:.4f} seconds")


   # Evaluate the trained agent

    eval_env = SatDynEnv()

    print("Evaluate model >>>")

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)

    print(f"mean_reward of trained agent:{mean_reward:.2f} +/- {std_reward:.2f}")

    print("Save model >>>")
    model.save("./models/ddpg_sat_gpu_temp_numba")


    # Visualizing training metrics
    plot_flag = 0
    if plot_flag == 1:
        # Arrays for storing data
        times = np.linspace(0, 300, 300 * 10)
        states = []
        torques = []

        kp = 50
        kd = 500

        env.render()
        while not done:
            # Sample a random action
            # action = env.action_space.sample()
            # action = torque_function(env.state, kp, kd)
            action, _states = model.predict(env.state)

            states.append(env.state.copy())
            torques.append(action.copy())

            # Step the environment
            next_state, reward, done, truncated, info = env.step(action)
            # Render the environment
            env.render()
        env.close()

        # Extract the solution for attitude (in terms of quaternion) and angular velocity
        states_array = np.array(states)
        torques_array = np.array(torques) * scale_torque
        q_0 = states_array[:, 0]
        q_1 = states_array[:, 1]
        q_2 = states_array[:, 2]
        q_3 = states_array[:, 3]
        omega_x = states_array[:, 4]
        omega_y = states_array[:, 5]
        omega_z = states_array[:, 6]

        # store the norm of quaternions
        norm_q = np.linalg.norm(states_array[:, :4], axis=1)

        # Plot the results
        fig = plt.figure(figsize=(12, 10))

        # Plot quaternion
        plt.subplot(3, 1, 1)
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
        plt.subplot(3, 1, 2)
        plt.plot(times, omega_x * (180 / np.pi), label='$\\omega_x$')
        plt.plot(times, omega_y * (180 / np.pi), label='$\\omega_y$')
        plt.plot(times, omega_z * (180 / np.pi), label='$\\omega_z$')
        plt.title('Angular velocity')
        plt.ylabel('$\\omega$ (deg/s)')
        plt.legend()
        plt.grid()

        # Plot torque input
        plt.subplot(3, 1, 3)
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
    else:
        pass


