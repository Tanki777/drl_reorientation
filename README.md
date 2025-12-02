# Spacecraft Reorientation Control with Pointing Constraint using SAC

## Setup
To use this repository, install the packages from ``requirements.txt``.
Make sure to install the correct CUDA version for PyTorch (for more details, see ``requirements.txt``).

## Project Structure
Initially, your directory structure should look like this:
````
└───drl_reorientation
    ├───agent_simulation
    └───agent_training
````
After using the trainer for the first time, it should look like this (``models_meta`` and ``schedules`` only appear after using the auto trainer for the first time):
````
├───drl_reorientation
│   ├───agent_simulation
│   └───agent_training
├───models
├───models_meta
├───monitor_logs
├───tensorboard
└───schedules
````

## Agent Training
### Environment
#### Content:
- Functions decorated with ``@jit`` for faster computation (does not use python)
- Action space, state space, reward function
- Environment class ``SatDynEnv``
- TensorBoard log generation
- Video frame generation

### Trainer
#### Content:
- Custom callback class for TensorBoard logging
- TensorBoard log saving
- TensorBoard server for log charts
- Vectorized environment (eight parallel environments) for more efficient training
- Log monitor for custom TensorBoard metrics
- Automatic model backups
- Training workflow
#### Usage:
- Parameters:
    - Set ``MODEL_NAME`` to the name of the model to be trained. Omit the file ending (i.e. .zip)
    - Set ``CONTINUE_TRAINING`` to ``False`` to create a new model with the given name and train it from scratch. Set it to ``True`` to instead load the latest model file with the given name to train it further. When set to ``True`` but the model file cannot be found, a new model with that name will be created and trained from scratch.
    - Set ``TRAINING_TIMESTEPS`` to an integer value. It defines how many timesteps the model will be trained when running the trainer. 1000 timesteps are equivalent to one episode in this setup.
    - Set ``CHECK_FREQ`` to an integer value. It defines after how many timesteps the custom callback function will log the custom TensorBoard metrics. Note that, since we use a vectorized environment with eight instances, a ``CHECK_FREQ`` of 1000 leads to logging after every 8000 timesteps. So, if ``CHECK_FREQ`` is set higher than 1/8 of ``TRAINING_TIMESTEPS``, TensorBoard will not log the custom metrics.
    - Set ``SAVE_INTERVAL`` to an integer value. It defines after how many total timesteps the model will be saved as a backup. This corresponds to the ``TRAINING_TIMESTEPS`` and is therefore not affected by the number of environments. Note that this interval follows the total timesteps tracked by the model. E.g. if a model starts training at 150,000 timesteps and ``TRAINING_TIMESTEPS`` is set to 100,000, a backup will be made at model timestep 200,000 if ``SAVE_INTERVAL`` is set to 100,000.
- Training:
    - During training, a progress bar will indicate how long the current training session is running and when it will be done.
    - After training, the model will be saved in ``models/MODEL_NAME_latest.zip`` where ``_latest`` is added as a suffix and automatically added when loading the model. Additionally, a backup model is saved to ``models/MODEL_NAME_TOTAL_TIMESTEPS_TIMESTAMP.zip``.
- TensorBoard:
    - At the beginning of the training session, a TensorBoard server is started and can be accessed via http://localhost:6006.
    - The UI will show all log folders located in the TensorBoard log directory.
    - When the training session is done, the TensorBoard server remains open until closed by pressing ``Ctrl+C`` in the terminal.

### Auto Trainer
#### Content:
- Automate agent training according to a predefined schedule
#### Usage:
- Parameters:
    - Set ``SCHEDULE_FILE_NAME`` to the name of the schedule to use. Must end with ``.json``
    - Set ``CONTINUE_TRAINING`` to ``False`` to create a new model and train it from scratch using the schedule. If set to ``True``, an already existing model can be loaded to continue training. Note that this only works if the model was previously trained with the auto trainer (according to a schedule) and that the same schedule must be used for a model. If not adhering, the model will not be representative anymore for comparison.
    - Set ``MODEL_NAME`` to the name of the model to be trained. Omit the file ending (i.e. .zip). Note that the suffix ``_latest`` is automatically used when loading the model, so do not include it in ``MODEL_NAME``.
- Schedule:
    - A schedule is a JSON file and defines different training phases. Each phase can have its own amount of training timesteps and initial state.
    - Example:
    ````json
    {
        "name": "schedule 1",
        "phases": [
            {
                "phase_name": "Test Phase 1: Easy",
                "timesteps": 10000,
                "min_initial_error_angle": 0.0,
                "max_initial_error_angle": 30.0,
                "min_initial_angular_velocity": 0.0,
                "max_initial_angular_velocity": 0.1
            },
            {
                "phase_name": "Test Phase 1: Medium",
                "timesteps": 20000,
                "min_initial_error_angle": 0.0,
                "max_initial_error_angle": 60.0,
                "min_initial_angular_velocity": 0.0,
                "max_initial_angular_velocity": 0.1
            },
            {
                "phase_name": "Test Phase 1: Hard",
                "timesteps": 30000,
                "min_initial_error_angle": 0.0,
                "max_initial_error_angle": 120.0,
                "min_initial_angular_velocity": 0.0,
                "max_initial_angular_velocity": 0.1
            }
        ]
    }
    ````
- TensorBoard:
    - When the auto trainer starts, a TensorBoard server is started and can be accessed via http://localhost:6006.
    - The UI will show all log folders located in the TensorBoard log directory.
    - When the auto trainer is done, the TensorBoard server remains open until closed by pressing ``Ctrl+C`` in the terminal.
- Training:
    - For each phase a separate training session is used.
    - At the end of each phase, the model is saved (with backup) and the metadata updated.
    - From the metadata, the auto trainer knows if a phase is already fully or partially completed and skips the phase or trains the remaining timesteps respectively.

### Visualization
#### Content:
- Evaluation environment call
- Video of 3D trajectory
- Plot of 3D trajectory, attitude angle, reward, attitude quaternion, angular velocity, torque
- Console results
#### Usage:
- Parameters:
    - Set ``MODEL_NAME`` to the name of the model to be visualized. Omit the file ending (i.e. .zip) and make sure the mode file is in the ``models`` directory.
    - Set ``INITIAL_STATE`` to define the initial state for the evaluation environment.
- Plots:
    - A window containing the plots will open.
    - After closing the window, a result summary will be printed in the terminal.
- Video:
    - A video called ``trajectory.mp4`` will be created. To keep it, move it to another directory or rename it. Otherwise, it will be overwritten next time.