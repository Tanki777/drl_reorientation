import json
import os
import time

import trainer

parent_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(parent_dir)
repo_parent_dir = os.path.dirname(repo_dir)
schedule_dir = os.path.join(repo_parent_dir, "schedules")
meta_dir = os.path.join(repo_parent_dir, "models_meta")

# Terminal colors
RED_START = trainer.RED_START
GREEN_START = trainer.GREEN_START
YELLOW_START = trainer.YELLOW_START
COLOR_END = trainer.COLOR_END

# Create schedules directory if it doesn't exist
if not os.path.exists(schedule_dir):
    os.makedirs(schedule_dir)


def load_schedule(schedule_file_name):
    schedule_file = os.path.join(schedule_dir, schedule_file_name)
    try:
        with open(schedule_file, 'r') as f:
            schedule = json.load(f)
        print("--------------------------------------------")
        print(f"|---{GREEN_START}Loaded schedule.{COLOR_END}")
    except FileNotFoundError:
        print(f"{RED_START}Schedule file not found: {schedule_file}.{COLOR_END}")
        exit(1)

    return schedule


def load_model_metadata(model_name):
    """
    Try to load model metadata file. If not found, exit the program.
    Args:
        model_name: Name of the model whose metadata to load
    Returns:
        metadata: Loaded metadata dictionary
    """
    metadata_file = os.path.join(meta_dir, f"{model_name}.meta")
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print("|")
        print(f"|---{GREEN_START}Loaded model metadata.{COLOR_END}")
    except FileNotFoundError:
        print("|")
        print(f"|---{RED_START}Metadata file not found: {metadata_file}.{COLOR_END}")
        exit(1)

    return metadata


def create_model_metadata(model_name, schedule):
    # Create meta header
    metadata = {
        "model_name": model_name,
        "schedule_name": schedule.get("schedule_name", ""),
    }

    # Parse phases from schedule
    phases = []
    for phase in schedule.get("phases", []):
        phases.append({
            "phase_name": phase.get("phase_name", ""),
            "timesteps_left": phase.get("timesteps", 0)
        })

    metadata["phases"] = phases

    return metadata


def save_model_metadata(model_name, metadata):
    # Save metadata to file
    metadata_file = os.path.join(meta_dir, f"{model_name}.meta")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)


def create_or_load_model(continue_training, model_name, env):
    # Load model and metadata
    if continue_training:
        model, save_path, latest_model_path = trainer.create_or_load_model(env, continue_training, model_name, trainer.log_path)

    # Create new model and metadata
    else:
        model, save_path, latest_model_path = trainer.create_or_load_model(env, continue_training, model_name, trainer.log_path)

    return model, save_path, latest_model_path


def do_scheduled_training(model_name, schedule, continue_training):
    # Get training phases
    phases = schedule.get("phases", [])
    phase_count = len(phases)

    # Get metadata
    if continue_training:
        metadata = load_model_metadata(model_name)
    else:
        metadata = create_model_metadata(model_name, schedule)

    # Perform training for each phase
    for phase_index, phase in enumerate(phases):
        # After the second phase, always continue training
        if phase_index >= 1:
            continue_training = True

        phase_name = phase.get("phase_name", "")
        timesteps = phase.get("timesteps", 0)
        initial_state = [
            phase.get("min_initial_error_angle", 0.0),
            phase.get("max_initial_error_angle", 30.0),
            phase.get("min_initial_angular_velocity", 0.0),
            phase.get("max_initial_angular_velocity", 0.1),
            phase.get("max_steps", 500),
            0.0,
            0.0
        ]
        timesteps_left = metadata["phases"][phase_index].get("timesteps_left", timesteps)

        if timesteps <= 0:
            print("|")
            print(f"|---{RED_START}Skipping phase {phase_name}, timesteps must be more than 0.{COLOR_END}")
            continue

        if timesteps_left <= 0:
            print("|")
            print(f"|---{YELLOW_START}Phase {phase_name} already done.{COLOR_END}")
            continue
        
        print("|")
        print(f"|---{YELLOW_START}Starting phase {phase_index+1}/{phase_count}:{COLOR_END}")
        for key, value in phase.items():
            print(f"|-----{key}: {value}")
        
        print("|")
        print(f"|---Training for {timesteps_left} timesteps...")

        # Create the training environment
        env = trainer.create_environment(model_name, initial_state=initial_state, phase_name=phase_name)

        # Create or load the model based on CONTINUE_TRAINING
        model, save_path, latest_model_path = create_or_load_model(continue_training, model_name, env)

        # Train the agent model for the specified timesteps
        model = trainer.train_agent(model, save_path, timesteps_left, 500, 100_000, model_name)
        print("|")
        print(f"|---{GREEN_START}Phase {phase_name} completed.{COLOR_END}")

        # Save the trained model
        trainer.save_model(model, model_name)

        # Update metadata
        metadata["phases"][phase_index]["timesteps_left"] = 0
        save_model_metadata(model_name, metadata)
        print(f"|-----{GREEN_START}Metadata saved.{COLOR_END}")
        

if __name__ == "__main__":
    # Define which schedule to use
    SCHEDULE_FILE_NAME = "test_schedule_old_env_1.json"
    CONTINUE_TRAINING = False
    MODEL_NAME = "test_new_env_rw_yang_old_sat_sched1"

    # Load the selected schedule
    schedule = load_schedule(SCHEDULE_FILE_NAME)

    # Monitor training progress in TensorBoard
    tensorboard_process = trainer.start_tensorboard()

    # Perform scheduled training
    do_scheduled_training(MODEL_NAME, schedule, CONTINUE_TRAINING)

    # Stop TensorBoard server on ctrl+C
    try:
        print("|")
        print(f"|---{YELLOW_START}Press Ctrl+C to stop the TensorBoard server.{COLOR_END}")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        trainer.stop_tensorboard(tensorboard_process)