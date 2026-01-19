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
    # 1. Init Environment
    # For playback, we want "human" render mode to see it in a window
    render_mode = "human"
    env = PongEnv(render_mode=render_mode)
    
    # We still need to wrap it for the AI to understand inputs
    # But we want to play the unwrapped env to see it? 
    # The wrappers modify the env.step() and observation space.
    # If we wrap it, 'env.render()' should still work if wrappers delegate it.
    
    # Re-create the wrapping stack manually since we aren't using make_vec_env with 'human' easily
    # Or just wrap the single env instance.
    env = CompatiblePongWrapper(env)
    
    # FrameStack is usually part of VecEnv in SB3.
    # To run a single instance with FrameStack, we can use gymnasium.wrappers.FrameStack
    # But SB3's VecFrameStack is what the model expects (batch dim).
    # Simplest way: Put it in a DummyVecEnv, then access the internal env for rendering if needed?
    # Actually, if we use DummyVecEnv([lambda: env]), the render() might be hidden.
    
    # Better approach for visualization:
    # Use the same setup as training but with render_mode='human' passed to env constructor.
    # DummyVecEnv supports simple rendering if we call env.render().
    
    env = DummyVecEnv([lambda: PongEnv(render_mode="human")])
    env = VecFrameStack(env, n_stack=4, channels_order='last')
    # Note: We missed CompatiblePongWrapper in the lambda above! Correcting...
    
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
    
    # 4. Loop
    obs = env.reset()
    
    running = True
    while running:
        # Agent Action
        action, _ = model.predict(obs, deterministic=True)
        
        # In Human vs AI mode, typically Human plays the "Agent" paddle?
        # Specification says: "Human vs AI: I play against the AI"
        # Our Environment: Agent = Right Paddle. Opponent = Left Paddle.
        # If User wants to play, User should control Right Paddle? Or Left?
        # Usually User = Player 1 (Left). AI = Player 2 (Right).
        # Our logic: Agent controls Right. Opponent controls Left.
        # So "Human vs AI" means Human is Left (Opponent) and AI is Right (Agent).
        # But our Env auto-moves the Opponent (Left).
        # We need to disable Opponent AI if Human is playing!
        
        if mode == "human":
            # We need to override the Internal Opponent Logic to be Keyboard controlled
            # This is tricky because the Env logic is hardcoded in step()
            # We can hack it or add a flag to PongEnv.
            
            # Let's handle keyboard events here and force the paddle position?
            # Or better: Add a 'human_opponent' flag to env.
            pass # See below
            
        # Step
        obs, rewards, dones, infos = env.step(action)
        
        # Handle User Input for Human Mode
        # We can detect keys here (Pygame is running)
        if mode == "human":
            keys = pygame.key.get_pressed()
            # We need to send this to the env. 
            # env.env_method("set_human_action", ...)
            human_action = 0
            if keys[pygame.K_w]: human_action = 1 # Up
            if keys[pygame.K_s]: human_action = 2 # Down
            
            env.env_method("set_opponent_action", human_action)
            
        time.sleep(1/60.0) # Cap FPS roughly

        if dones[0]:
            print("Round Over")
            obs = env.reset()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model zip")
    parser.add_argument("--opponent", type=str, default=None, help="Path to opponent model zip (optional)")
    parser.add_argument("--mode", type=str, default="human", choices=["human", "ai_vs_ai"], help="Mode")
    args = parser.parse_args()

    play(args.model, args.mode, args.opponent)
