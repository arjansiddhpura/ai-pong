import argparse
import os
import torch
import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
# Note: AtariWrapper not needed since we use custom PongEnv with CompatiblePongWrapper
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation

from pong_env import PongEnv

# --- Hardware Setup ---
def get_device():
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("Using Apple MPS (Metal Performance Shaders)")
        return "mps"
    else:
        print("Using CPU")
        return "cpu"

# --- Wrappers ---
class CompatiblePongWrapper(gym.Wrapper):
    """
    Custom wrapper to ensure compatibility with SB3 CnnPolicy.
    Scales observations to 84x84 and Grayscale.
    """
    def __init__(self, env):
        env = ResizeObservation(env, (84, 84))
        env = GrayscaleObservation(env, keep_dim=True) # (84, 84, 1)
        # Note: SB3 VecFrameStack expects the env to output channels-last if it's an image
        # but internal processing often handles (C, H, W) conversion.
        # However, for PPO CnnPolicy, we want input to be normalized 0-1 usually, but SB3 does that internally if configured.
        super().__init__(env)

def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.
    Each environment gets a unique seed based on rank.
    """
    def _init():
        env = PongEnv()
        env = CompatiblePongWrapper(env)
        env = Monitor(env)  # Record stats
        env.reset(seed=seed + rank)
        return env
    return _init

# --- Self-Play Callback ---
class SelfPlayCallback(BaseCallback):
    """
    Callback for self-play.
    Periodically saves the current agent and updates the opponent to use this agent.
    """
    def __init__(self, update_interval, save_path, verbose=1):
        super().__init__(verbose)
        self.update_interval = update_interval
        self.save_path = save_path
        self.generation = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.update_interval == 0:
            if self.verbose > 0:
                print(f"Self-Play Update: Generation {self.generation}")
            
            # Save current model
            model_name = os.path.join(self.save_path, f"opponent_gen_{self.generation}")
            self.model.save(model_name)
            
            # Update Opponent Logic in Environment
            # Since we are using VecEnv, we need to call a method on the wrapped envs.
            # This is tricky with SubprocVecEnv.
            # Simplified approach: We just print here, but in a real setup we need to pass the path back to the envs.
            # For now, let's assume valid logic for single process or share memory.
            
            # TODO: Ideally, we reload the model in the env.
            # `self.training_env` is the VecEnv.
            # We can use `env_method` to call a function on each env.
            self.training_env.env_method("update_opponent_model", model_name + ".zip")
            
            self.generation += 1
        return True

# Add this method to PongEnv in pong_env.py to support the callback
# def update_opponent_model(self, model_path):
#    self.opponent_policy = PPO.load(model_path)


# --- Main Training Loop ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total training steps")
    parser.add_argument("--update_freq", type=int, default=50_000, help="Steps before updating opponent")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--perfect_opponent", action="store_true", help="Train against perfect AI (no self-play)")
    args = parser.parse_args()

    device = get_device()
    save_dir = "./models"
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Create Vectorized Environment
    # Use SubprocVecEnv for true parallelism on cluster
    # DummyVecEnv is used for Mac/MPS since SubprocVecEnv has issues with MPS
    if device == "mps":
        # Mac: Use DummyVecEnv for compatibility
        env = DummyVecEnv([make_env(i) for i in range(args.n_envs)])
    else:
        # Linux/CUDA: Use SubprocVecEnv for true parallelism
        env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
    
    # 2. Frame Stacking
    # Stack 4 frames
    env = VecFrameStack(env, n_stack=8, channels_order='last') 

    # 3. Define Model
    # CnnPolicy is standard for pixel inputs
    model = PPO(
        "CnnPolicy", 
        env, 
        device=device,
        verbose=1, 
        tensorboard_log="./logs",
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=256,
        n_epochs=4,
        ent_coef=0.01
    )
    
    # 4. Callback - only use self-play if not training against perfect opponent
    if args.perfect_opponent:
        print("Training against PERFECT OPPONENT (no self-play)")
        callback = None
    else:
        callback = SelfPlayCallback(update_interval=args.update_freq, save_path=save_dir)

    # 5. Train
    print(f"Starting training on {device} for {args.steps} steps...")
    model.learn(total_timesteps=args.steps, callback=callback)
    
    # 6. Save Final Model
    model.save("final_pong_agent")
    env.close()
    print("Training Complete.")
