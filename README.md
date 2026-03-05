# Spacecraft Reorientation Control with Pointing Constraint using SAC

## Setup
To use this repository, install the packages from ``requirements.txt``.
Make sure to install the correct CUDA version for PyTorch (for more details, see ``requirements.txt``).

## Project Structure
Initially, your directory structure should look like this:
````
└───drl_reorientation
    ├───agent_simulation
    ├───agent_training
    └───safety_filter
````
After using the auto trainer, evaluation and visualization tools for the first time, it should look like this:
````
├───drl_reorientation
│   ├───agent_simulation
│   ├───agent_training
│   └───safety_filter
├───evaluation_data
├───models
├───models_meta
├───models_replay_buffers
├───monitor_logs
├───schedules
├───tensorboard
└───videos
````

## Agent Training
### Environment
#### Content:
- Functions decorated with ``@jit`` for faster computation (does not use python)
- Action space, state space, reward function
- Environment class ``SatDynEnv``
- TensorBoard log generation
- Video frame generation

### Auto Trainer
#### Content:
- Automate agent training according to a predefined schedule
- Models and replay buffers are saved every 100k timesteps by default
#### Usage:
- Parameters:
    - Set ``SCHEDULE_FILE_NAME`` to the name of the schedule to use. Must end with ``.json``
    - Set ``CONTINUE_TRAINING`` to ``False`` to create a new model and train it from scratch using the schedule. If set to ``True``, an already existing model can be loaded to continue training. Note that this only works if the model was previously trained with the auto trainer (according to a schedule) and that the same schedule must be used for a model. If not adhering, the model will not be representative anymore for comparison.
    - Set ``USE_SAFETY_FILTER`` to ``0`` if no filter should be applied. If set to ``1``, it is applied after training (only useful for simulation / evaluation). If set to ``2``, it is applied during training.
    - Set ``MODEL_NAME`` to the name of the model to be trained. Omit the file ending (i.e. .zip). Note that the suffix ``_latest`` is automatically used when loading the model, so do not include it in ``MODEL_NAME``.
- Schedule:
    - A schedule is a JSON file and defines different training phases. Each phase can have its own amount of training timesteps and initial state.
    - If training for the first time, create a ``schedules`` folder if it does not exist and put your schedule file in there. For the location of the folder, see Section ``Project Structure``.
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
                "max_initial_angular_velocity": 0.1,
                "max_steps": 3000,
                "min_half_angle_koz": 0.0,
                "max_half_angle_koz": 0.0
            },
            {
                "phase_name": "Test Phase 1: Medium",
                "timesteps": 20000,
                "min_initial_error_angle": 0.0,
                "max_initial_error_angle": 180.0,
                "min_initial_angular_velocity": 0.0,
                "max_initial_angular_velocity": 0.1,
                "max_steps": 3000,
                "min_half_angle_koz": 0.0,
                "max_half_angle_koz": 0.0
            },
            {
                "phase_name": "Test Phase 2: Hard",
                "timesteps": 30000,
                "min_initial_error_angle": 80.0,
                "max_initial_error_angle": 180.0,
                "min_initial_angular_velocity": 0.0,
                "max_initial_angular_velocity": 0.1,
                "max_steps": 3000,
                "min_half_angle_koz": 15.0,
                "max_half_angle_koz": 30.0
            }
        ]
    }
    ````
- TensorBoard:
    - When the auto trainer starts, a TensorBoard server is started and can be accessed via http://localhost:6006.
    - The UI will show all log folders located in the TensorBoard log directory.
    - For auto refreshing new logs, click the gear icon in the upper right corner (the one left to the encircled ``?`` icon). Then tick ``Reload data`` and set the ``Reload period`` to e.g. 60.
    - When the auto trainer is done, the TensorBoard server remains open until closed by pressing ``Ctrl+C`` in the terminal.
    - When not training, the TensorBoard server can be started with ``tensorboard_server_standalone.py`` to access the logs.
- Training:
    - For each phase a separate training session is used.
    - At the end of each phase, the model is saved (with backup) and the metadata updated.
    - From the metadata, the auto trainer knows if a phase is already fully or partially completed and skips the phase or trains the remaining timesteps respectively.
    - If the training is disrupted before the end of a schedule phase, the latest model and replay buffer files are not updated and are still from the last phase. In that case, one can manually rename the most recently saved files to end with ``_latest`` if one wants to continue training from that point.
    - If a training session is done and should be continued, but the schedule has no more phases, one can edit the metadata file of the model and set the timesteps left to the new desired amount of training timesteps to continue training without using a new schedule.

## Agent Simulation
### Environment Simulator
#### Content:
- Simulate the satellite dynamics
#### Usage:
- Parameters:
    - Set ``INITIAL_STATE`` to define the initial state of the satellite.
- Action schedule:
    - A predefined sequence of actions is simulated. Edit ``action_schedule(t)`` to define which actions should be performed at which time.

### Evaluation
#### Content:
- Calculate metrics for multiple episodes
- Monte Carlo simulation
#### Usage:
- Parameters:
    - Set ``MODEL_NAME`` to the name of the model to evaluate. Omit the file ending (i.e. .zip) and make sure the mode file is in the ``models`` directory.
    - Set ``MAX_STEPS`` to the number of time steps an episode should have.
    - Set ``INITIAL_STATE`` to define the initial state for the evaluation environment.
    - Set ``USE_SAFETY_FILTER`` to ``0`` if no filter should be applied (if the model was trained with filter and is supposed to be evaluated without applying the filter after training additionally, set it to ``0`` too). If set to ``1``, it is applied after training (if the model was trained with filter, using ``1`` does not include the reward penalty for differing agent and filter action). If set to ``2``, it is applied after training and the reward will include the penalty for differing agent and filter action.
- Calculating metrics:
    - Uncomment the corresponding section in main to load evaluation data and calculate metrics for all episodes.
- Monte Carlo simulation:
    - Uncomment the corresponding section in main to perform the simulation.
    - Set ``episodes`` to the total number of episodes to simulate.
    - Set ``num_workers`` to the number of CPU cores to use. It is recommended to use 1 worker for a model without safety filter applied after training, and 4 workers if applied after training (higher worker numbers showed no increased speed due to GPU bottleneck).

### Visualization
#### Content:
- Evaluation environment call
- Video of 3D trajectory
- Plot of 3D trajectory, attitude angle, reward, attitude quaternion, angular velocity, torque
- Console results
#### Usage:
- Parameters:
    - Set ``MODEL_NAME`` to the name of the model to be visualized. Omit the file ending (i.e. .zip) and make sure the mode file is in the ``models`` directory.
    - Set ``MAX_STEPS`` to the number of time steps an episode should have.
    - Set ``INITIAL_STATE`` to define the initial state for the evaluation environment.
    - Set ``CREATE_VIDEO`` to ``true`` if a .mp4 of the boresight trajectory should be captured. This slows down simulation speed noticably.
    - Set ``USE_SAFETY_FILTER`` to ``0`` if no filter should be applied (if the model was trained with filter and is supposed to be evaluated without applying the filter after training additionally, set it to ``0`` too). If set to ``1``, it is applied after training (if the model was trained with filter, using ``1`` does not include the reward penalty for differing agent and filter action). If set to ``2``, it is applied after training and the reward will include the penalty for differing agent and filter action.
- Plots:
    - A regular plot can be created with ``plot_actual_attitude(simulation_data)``, which is preferred for agent evaluation.
    - A reduced plot optimized for papers can be created with ``plot_for_report(simulation_data,time_end=300)``, where ``time_end`` is the maximum time in seconds which is plotted for the metrics (to cut the plot after settling). The 3D trajectory plot is not affected by this.
- Video:
    - If the parameter is set, a timestamped video of the trajectory will be created. The video always uses 30 fps by default.
