import os
import time
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from test_environment import SatDynEnv
import subprocess

# Get the log and models path
parent_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(parent_dir)
repo_parent_dir = os.path.dirname(repo_dir)
log_path = os.path.join(repo_parent_dir, "sac_sat_gpu_temp_numba_tensorboard")
models_path = os.path.join(repo_parent_dir, "models")

class CustomCallback(BaseCallback):
    def __init__(self, check_freq, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.env_interactions = 0  # Track total environment interactions
        
        # Custom metrics accumulators
        self.custom_metrics = {
            "initial_error_angle": [],
            "initial_angular_velocity": [],
            "final_error_angle": [],
            "settling_time": [],
            "avg_torque": [],
            "max_torque": [],
            "settled": []
        }

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
            
        # Collect custom metrics from episode endings
        for info in infos:
            if isinstance(info, dict):
                # Check if this info contains custom metrics (episode ended)
                has_custom_metrics = any(key.startswith("custom_metrics/") for key in info.keys())
                if has_custom_metrics:
                    for metric_name in self.custom_metrics.keys():
                        metric_key = f"custom_metrics/{metric_name}"
                        if metric_key in info:
                            self.custom_metrics[metric_name].append(info[metric_key])

        # Log metrics periodically
        if self.n_calls % self.check_freq == 0:
            self._log_custom_metrics()
            
        return True
    
    def _log_custom_metrics(self):
        """Log accumulated custom metrics to TensorBoard"""
        for metric_name, values in self.custom_metrics.items():
            if values:  # Only log if we have data
                mean_value = sum(values) / len(values)
                # Log to TensorBoard using the logger
                self.logger.record(f"custom/{metric_name}_mean", mean_value)
                
                # For settled episodes, also log success rate
                if metric_name == 'settled':
                    success_rate = mean_value  # settled contains 0/1 values
                    self.logger.record(f"custom/success_rate", success_rate)
                
                # Clear the accumulated values
                self.custom_metrics[metric_name] = []


def start_tensorboard():
    """Start TensorBoard server in background, access with http://localhost:6006"""
    print(f"Looking for TensorBoard logs in: {log_path}")

    # Check if the log directory exists
    if not os.path.exists(log_path):
        print(f"Log directory does not exist: {log_path}")
        print("Available directories:")
        for item in os.listdir(repo_dir):
            item_path = os.path.join(repo_dir, item)
            if os.path.isdir(item_path) and "tensorboard" in item.lower():
                print(f"{item}")
    else:
        print(f"Log directory found: {log_path}")

    try:
        print("Starting TensorBoard server...")
        
        # Start TensorBoard process in background
        process = subprocess.Popen([
            "tensorboard",
            f"--logdir={log_path}",
            "--port=6006",
            "--host=localhost"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("TensorBoard server starting...")
        print("Access TensorBoard at: http://localhost:6006")
        print("Press Ctrl+C to stop the server")
        
        # Wait a moment for TensorBoard to start
        time.sleep(3)

        return process
    
    except FileNotFoundError:
        print("TensorBoard not found.")


def stop_tensorboard(process):
    """Stop TensorBoard server"""
    if process is not None:
        try:
            print("Stopping TensorBoard server...")
            process.terminate()
            process.wait(timeout=5)
            print("TensorBoard server stopped")
        except subprocess.TimeoutExpired:
            print("Force killing TensorBoard server...")
            process.kill()
            process.wait()
            print("TensorBoard server force stopped")
        except Exception as e:
            print(f"Error stopping TensorBoard: {e}")


if __name__ == "__main__":

    print("Creating environment...", end="")
    
    # Create monitor directory for episode logging
    monitor_dir = os.path.join(repo_parent_dir, "monitor_logs")
    if not os.path.exists(monitor_dir):
        os.makedirs(monitor_dir)

    # Create vectorized environment with 8 parallel instances
    env = make_vec_env(SatDynEnv, n_envs=8)
    
    # Add timestamp to monitor file to prevent overwriting previous runs
    timestamp = int(time.time())
    monitor_log_file = os.path.join(monitor_dir, f"training_monitor_{timestamp}")
    
    # Track custom metrics in VecMonitor
    custom_info_keywords = (
        "custom_metrics/initial_error_angle",
        "custom_metrics/initial_angular_velocity", 
        "custom_metrics/final_error_angle",
        "custom_metrics/settling_time",
        "custom_metrics/avg_torque",
        "custom_metrics/max_torque",
        "custom_metrics/settled",
    )
    
    # Wrap environment with VecMonitor to log episode info
    env = VecMonitor(env, filename=monitor_log_file, info_keywords=custom_info_keywords)
    
    print(" done.")
   
    # Training configuration
    CONTINUE_TRAINING = True  # Set to True to load existing model, False for fresh start
    MODEL_NAME = "sac_sat_faster_2"  # Base name for saved models
    TRAINING_TIMESTEPS = 20_000  # Number of timesteps per training session
    CHECK_FREQ = 1_000  # Frequency of callback checks

    # Ensure the directory exists
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    # Setting the path to save the model
    save_path = os.path.join(models_path, MODEL_NAME)
    latest_model_path = os.path.join(models_path, f"{MODEL_NAME}_latest.zip")

    print("Creating/Loading the agent.")
    
    # Try to load existing model if CONTINUE_TRAINING is True
    if CONTINUE_TRAINING and os.path.exists(latest_model_path):
        print(f"Loading existing model from: {latest_model_path}")

        try:
            model = SAC.load(latest_model_path, device='cuda')
            model.set_env(env)
            print("Successfully loaded existing model!")
            print(f"Previous total timesteps: {model.num_timesteps}")
            
            # Update tensorboard log directory to continue logging
            model.tensorboard_log = log_path
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Creating new model instead...")
            CONTINUE_TRAINING = False
    
    # Create new model if not loading existing one
    if not CONTINUE_TRAINING or not os.path.exists(latest_model_path):
        print("Creating new model from scratch...")
        model = SAC("MlpPolicy", env, batch_size=2048, gradient_steps=8, policy_kwargs=dict(
        net_arch=dict(pi=[512, 512], qf=[512, 512])), verbose=1, device='cuda',
                    tensorboard_log=log_path)  # Use absolute path for consistency
        
    custom_callback = CustomCallback(check_freq=CHECK_FREQ)
    
    # Monitor training progress in TensorBoard
    tensorboard_process = start_tensorboard()
    
    print("Start training the agent >>> ")
    start_time = time.time()
    
    model.learn(total_timesteps=TRAINING_TIMESTEPS, progress_bar=True, callback=custom_callback, tb_log_name="SAC_47", reset_num_timesteps=False)
    end_time = time.time()
    
    print(f"Training time: {end_time - start_time:.2f} seconds")
    print(f"Current total timesteps: {model.num_timesteps}")

    # Save the updated model
    print("Saving improved model...")
    
    # Save with timestamp for history
    timestamp = int(time.time())
    timestamped_path = f"{save_path}_{model.num_timesteps}_{timestamp}"
    model.save(timestamped_path)
    
    # Save as latest model (for next session)
    model.save(latest_model_path.replace('.zip', ''))  # Remove .zip as save() adds it
    
    print(f"Model saved to:")
    print(f"    Latest: {latest_model_path}")
    print(f"    Backup: {timestamped_path}.zip")

    # Stop TensorBoard server on ctrl+C
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_tensorboard(tensorboard_process)