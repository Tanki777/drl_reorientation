import os
import time
import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
from test_environment import SatDynEnv
import subprocess

# Terminal colors
RED_START = "\033[91m"
GREEN_START = "\033[92m"
YELLOW_START = "\033[93m"
COLOR_END = "\033[0m"

# Get the log and models path
parent_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(parent_dir)
repo_parent_dir = os.path.dirname(repo_dir)
log_path = os.path.join(repo_parent_dir, "tensorboard")
if not os.path.exists(log_path):
    os.makedirs(log_path)
models_path = os.path.join(repo_parent_dir, "models")

class CustomCallback(BaseCallback):
    def __init__(self, check_freq, save_interval, base_save_path, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_interval = save_interval
        self.base_save_path = base_save_path
        
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

    def _on_training_start(self):
        # Define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        hparam_dict = {
                "algorithm": self.model.__class__.__name__,
                "learning rate": self.model.learning_rate,
                "tau": self.model.tau,
                "gamma": self.model.gamma,
        }
        
        # Tensorbaord will find & display metrics from the SCALARS tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0.0
        }
        
        
        self.logger.record("hparams", HParam(hparam_dict, metric_dict), exclude=("stdout", "log", "json", "csv"))

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

        # Save model every save_interval total timesteps
        if self.num_timesteps % self.save_interval == 0:
            # Create a unique model filename using timestamp
            timestamp = int(time.time())
            unique_model_path = f"{self.base_save_path}_{self.num_timesteps}_{timestamp}"

            # Save the model
            self.model.save(unique_model_path)
            
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
    print("|")
    print(f"|---{YELLOW_START}Looking for TensorBoard logs in: {log_path}{COLOR_END}")

    # Check if the log directory exists
    if not os.path.exists(log_path):
        print(f"|-----{RED_START}Log directory does not exist: {log_path}{COLOR_END}")
        print(f"|-----{RED_START}Available directories:{COLOR_END}")
        for item in os.listdir(repo_dir):
            item_path = os.path.join(repo_dir, item)
            if os.path.isdir(item_path) and "tensorboard" in item.lower():
                print(f"|-------{RED_START}{item}{COLOR_END}")
    else:
        print(f"|-----{GREEN_START}Log directory found: {log_path}{COLOR_END}")

    try:
        print("|")
        print(f"|---{YELLOW_START}Starting TensorBoard server...{COLOR_END}")
        
        # Start TensorBoard process in background
        process = subprocess.Popen([
            "tensorboard",
            f"--logdir={log_path}",
            "--port=6006",
            "--host=localhost"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("|-----Access TensorBoard at: http://localhost:6006")
        print("|-----Press Ctrl+C to stop the server")
        
        # Wait a moment for TensorBoard to start
        time.sleep(3)

        return process
    
    except FileNotFoundError:
        print(f"|-----{RED_START}TensorBoard not found.{COLOR_END}")


def stop_tensorboard(process):
    """Stop TensorBoard server"""
    if process is not None:
        try:
            print(f"|---{YELLOW_START}Stopping TensorBoard server...{COLOR_END}")
            process.terminate()
            process.wait(timeout=5)
            print(f"|-----{GREEN_START}TensorBoard server stopped{COLOR_END}")
        except subprocess.TimeoutExpired:
            print(f"|-----{RED_START}Force killing TensorBoard server...{COLOR_END}")
            process.kill()
            process.wait()
            print(f"|-----{RED_START}TensorBoard server force stopped{COLOR_END}")
        except Exception as e:
            print(f"|-----{RED_START}Error stopping TensorBoard: {e}{COLOR_END}")


def create_environment(initial_state=None):
    """
    Create the training environment
    Returns:
        env: The created and wrapped environment
    """
    print("--------------------------------------------")
    print(f"|---{YELLOW_START}Creating environment...{COLOR_END}")
    
    # Create monitor directory for episode logging
    monitor_dir = os.path.join(repo_parent_dir, "monitor_logs")
    if not os.path.exists(monitor_dir):
        os.makedirs(monitor_dir)

    # Create vectorized environment with 8 parallel instances
    if initial_state is not None:
        # Need to use a lambda to pass initial_state parameter
        env = make_vec_env(lambda: SatDynEnv(initial_state=initial_state), n_envs=8)
    else:
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

    return env


def create_or_load_model(env, continue_training, model_name, log_path):
    """
    Create a new SAC model or load an existing one depending on continue_training.
    Args:
        env: The training environment
        continue_training: Boolean indicating whether to continue training from an existing model
        model_name: Name of the model file
        log_path: Path for tensorboard logs
    Returns:
        model: The created or loaded SAC model
        save_path: Path where the model will be saved
        latest_model_path: Path to the latest saved model
    """
    # Ensure the directory exists
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    # Setting the path to save the model
    save_path = os.path.join(models_path, model_name)
    latest_model_path = os.path.join(models_path, f"{model_name}_latest.zip")

    print("|")
    print(f"|---{YELLOW_START}Creating/Loading the agent...{COLOR_END}")
    
    # Try to load existing model if CONTINUE_TRAINING is True
    if continue_training and os.path.exists(latest_model_path):
        print(f"|-----{YELLOW_START}Loading existing model from: {latest_model_path}{COLOR_END}")

        try:
            model = SAC.load(latest_model_path, device='cuda')
            model.set_env(env)
            print(f"|-----{GREEN_START}Successfully loaded existing model!{COLOR_END}")
            print(f"|-----Previous total timesteps: {model.num_timesteps}")
            
            # Update tensorboard log directory to continue logging
            model.tensorboard_log = log_path
            
        except Exception as e:
            print(f"|-----{RED_START}Failed to load model: {e}{COLOR_END}")
            print(f"|-----{YELLOW_START}Creating new model instead...{COLOR_END}")
            continue_training = False
    
    # Create new model if not loading existing one
    if not continue_training or not os.path.exists(latest_model_path):
        print(f"|-----{YELLOW_START}Creating new model from scratch...{COLOR_END}")
        model = SAC("MlpPolicy", env, batch_size=2048, gradient_steps=8, policy_kwargs=dict(
        net_arch=dict(pi=[512, 512], qf=[512, 512])), verbose=1, device='cuda',
                    tensorboard_log=log_path)  # Use absolute path for consistency
        
    return model, save_path, latest_model_path


def train_agent(model, save_path, total_timesteps, check_freq, save_interval, model_name):
    """
    Train the agent model with custom callback for logging and saving.
    Args:
        model: The SAC model to train
        save_path: Base path to save the model
        total_timesteps: Number of timesteps to train
        check_freq: Frequency of callback checks
        save_interval: Interval of timesteps to save the model
        model_name: Name of the model for tensorboard logging
    Returns:
        model: The trained SAC model
    """
    custom_callback = CustomCallback(check_freq=check_freq, save_interval=save_interval, base_save_path=save_path)

    print("|")
    print(f"|---{YELLOW_START}Start training the agent...{COLOR_END}")
    start_time = time.time()
    
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=custom_callback, tb_log_name=model_name, reset_num_timesteps=False)
    end_time = time.time()
    
    # Print training duration in a formatted way
    training_duration = datetime.timedelta(seconds=end_time - start_time)
    
    # Convert timedelta to datetime for formatting
    duration_datetime = datetime.datetime(1900, 1, 1) + training_duration
    formatted_duration = duration_datetime.strftime("%H:%M:%S")

    print(f"|-----Training completed in: {formatted_duration}")
    print(f"|-----Current total timesteps: {model.num_timesteps}")

    return model


def save_model(model, save_path, latest_model_path):
    """
    Save the trained model.
    Args:
        model: The trained SAC model
        save_path: Base path to save the model
        latest_model_path: Path to the latest saved model
    """
    # Save the updated model
    print("|")
    print(f"|---{YELLOW_START}Saving improved model...{COLOR_END}")
    
    # Save with timestamp for history
    timestamp = int(time.time())
    timestamped_path = f"{save_path}_{model.num_timesteps}_{timestamp}"
    model.save(timestamped_path)
    
    # Save as latest model (for next session)
    model.save(latest_model_path.replace('.zip', ''))  # Remove .zip as save() adds it
    
    print(f"|-----{GREEN_START}Model saved to:{COLOR_END}")
    print(f"|-------Latest: {latest_model_path}")
    print(f"|-------Backup: {timestamped_path}.zip")


if __name__ == "__main__":
    # Training configuration
    CONTINUE_TRAINING = True  # Set to True to load existing model, False for fresh start
    MODEL_NAME = "sac_sat_faster_2"  # Base name for saved models
    TRAINING_TIMESTEPS = 10_000  # Number of timesteps per training session
    CHECK_FREQ = 1_000  # Frequency of callback checks every CHECK_FREQ timesteps
    SAVE_INTERVAL = 200_000  # Model backup saved after every SAVE_INTERVAL timesteps

    # Create the training environment
    env = create_environment()

    # Create or load the agent model
    model, save_path, latest_model_path = create_or_load_model(env, CONTINUE_TRAINING, MODEL_NAME, log_path=log_path)
    
    # Monitor training progress in TensorBoard
    tensorboard_process = start_tensorboard()
    
    # Train the agent model
    model = train_agent(model, save_path, TRAINING_TIMESTEPS, CHECK_FREQ, SAVE_INTERVAL, MODEL_NAME)

    # Save the trained model
    save_model(model, save_path, latest_model_path)

    # Stop TensorBoard server on ctrl+C
    try:
        print("|")
        print(f"|---{YELLOW_START}Press Ctrl+C to stop the TensorBoard server.{COLOR_END}")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_tensorboard(tensorboard_process)