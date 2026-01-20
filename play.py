import argparse
import time
import gymnasium as gym
import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

from pong_env import PongEnv
from train import CompatiblePongWrapper  # Reuse wrapper logic

def play(model_path, mode="human", opponent_path=None):
    # 1. Create Environment with proper wrapping
    # We wrap in DummyVecEnv + VecFrameStack for compatibility with SB3 model
    def make_env():
        e = PongEnv(render_mode="human")
        e = CompatiblePongWrapper(e)
        return e
    
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4, channels_order='last')

    # 2. Load Agent
    print(f"Loading Agent from {model_path}")
    model = PPO.load(model_path)

    # 3. Load Opponent (if AI vs AI)
    # The env needs to know about the opponent.
    # We need to access the underlying PongEnv.
    # env is VecFrameStack -> DummyVecEnv -> CompatiblePongWrapper -> PongEnv
    # We can use env.env_method to call functionality 
    
    if opponent_path:
        print(f"Loading Opponent from {opponent_path}")
        # Pass the path to the environment so it can load it internally
        env.env_method("update_opponent_model", opponent_path)
    
    # Enable human mode if playing Human vs AI
    if mode == "human":
        env.env_method("enable_human_mode", True)
    
    # 4. Loop
    obs = env.reset()
    
    running = True
    while running:
        # CRITICAL: Process pygame events to prevent window freeze and enable keyboard input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Handle User Input for Human Mode BEFORE stepping
        # This ensures the human action is set before the environment processes it
        if mode == "human":
            keys = pygame.key.get_pressed()
            human_action = 0
            if keys[pygame.K_w]: human_action = 1  # Up
            if keys[pygame.K_s]: human_action = 2  # Down
            env.env_method("set_opponent_action", human_action)
        
        # Agent Action (AI controls the Right Paddle)
        action, _ = model.predict(obs, deterministic=True)
            
        # Step the environment
        obs, rewards, dones, infos = env.step(action)
            
        time.sleep(1/60.0)  # Cap FPS roughly

        if dones[0]:
            print("Round Over")
            obs = env.reset()
    
    # Cleanup
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model zip")
    parser.add_argument("--opponent", type=str, default=None, help="Path to opponent model zip (optional)")
    parser.add_argument("--mode", type=str, default="human", choices=["human", "ai_vs_ai"], help="Mode")
    args = parser.parse_args()

    play(args.model, args.mode, args.opponent)
