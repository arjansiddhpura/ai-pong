# Ping Pong RL Playground

This project implements a Reinforcement Learning agent to play Ping Pong from raw pixels using Stable-Baselines3 and PyGame. It features a self-play mechanism where the agent trains against checkpoints of itself.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running Locally (Apple Silicon)

### Training

To start training on your local machine (will use MPS acceleration if available):

```bash
python train.py --steps 100000 --n_envs 1
```

### Playback

To play against the trained AI (Human vs AI):

```bash
python play.py --model models/opponent_gen_0.zip --mode human
```

_(Note: Replace `models/opponent_gen_0.zip` with your latest model path)_

To watch AI vs AI:

```bash
python play.py --model models/opponent_gen_X.zip --opponent models/opponent_gen_Y.zip --mode ai_vs_ai
```

## Cluster Deployment (Headless)

The code is designed to detect if a display is available. If running on a cluster without a monitor, it will automatically switch to a "dummy" video driver for PyGame.

### Steps to Run on Cluster:

1. **Environment**: Ensure you have a virtual environment with the requirements installed.
    ```bash
    module load python/3.x # Example
    pip install -r requirements.txt
    ```
2. **Run Training**:
   Submit a job or run interactively. The script will automatically detect CUDA GPUs.

    ```bash
    python train.py --steps 10000000 --n_envs 8
    ```

    _Note: `n_envs` is set to 8 to utilize the 8-core/GPU power efficiently if using SubprocVecEnv (currently DummyVecEnv in code for stability, but parallelization can be enabled)._

3. **Monitor**:
   Logs are saved to `./logs` (TensorBoard) and models to `./models`.
    ```bash
    tensorboard --logdir ./logs
    ```

## Code Structure

- `pong_env.py`: Custom Gym Environment handling the game logic and pixel rendering.
- `train.py`: Main training script with Self-Play callback and PPO configuration.
- `play.py`: Script for visualization and human play.
