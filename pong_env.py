import gymnasium as gym
import numpy as np
import pygame
import cv2
import os
from gymnasium import spaces

class PongEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.screen_width = 800
        self.screen_height = 600
        
        # Action Space: 0 = Stay, 1 = Up, 2 = Down
        self.action_space = spaces.Discrete(3)
        
        # Observation Space: Raw pixels (Screen Height, Screen Width, 3 RGB channels)
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(self.screen_height, self.screen_width, 3), 
            dtype=np.uint8
        )

        # Game State
        self.paddle_width = 15
        self.paddle_height = 90
        self.ball_size = 15
        self.paddle_speed = 6
        self.ball_speed_val = 7
        
        # Pygame Setup
        if self.render_mode != "human":
             os.environ["SDL_VIDEODRIVER"] = "dummy"
        
        pygame.init()
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Pong RL")
        else:
            # Hidden surface for headless rendering
            self.screen = pygame.Surface((self.screen_width, self.screen_height))
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 74)

        # Opponent Policy (can be a function or a trained model)
        self.opponent_policy = None 
        
        # Human mode: When True, opponent is controlled by keyboard instead of AI
        self.human_mode = False
        self.human_action = 0  # 0=Stay, 1=Up, 2=Down

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset Positions
        self.left_paddle_y = self.screen_height // 2 - self.paddle_height // 2
        self.right_paddle_y = self.screen_height // 2 - self.paddle_height // 2
        
        # Reset Ball
        self.ball_x = self.screen_width // 2
        self.ball_y = self.screen_height // 2
        
        # Randomize ball direction
        angle = self.np_random.uniform(-np.pi/4, np.pi/4)
        direction = -1 if self.np_random.random() < 0.5 else 1 # Left or Right
        self.ball_vx = self.ball_speed_val * np.cos(angle) * direction
        self.ball_vy = self.ball_speed_val * np.sin(angle)
        
        # Only reset scores on first call or if explicitly requested
        # This allows scores to persist between rounds during gameplay
        reset_scores = options.get("reset_scores", False) if options else False
        if not hasattr(self, 'score_left') or reset_scores:
            self.score_left = 0
            self.score_right = 0
        
        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), {}

    def step(self, action):
        # 1. Handle Right Paddle (Agent)
        if action == 1: # Up
            self.right_paddle_y -= self.paddle_speed
        elif action == 2: # Down
            self.right_paddle_y += self.paddle_speed
        
        # Clamp paddle
        self.right_paddle_y = np.clip(self.right_paddle_y, 0, self.screen_height - self.paddle_height)

        # 2. Handle Left Paddle (Opponent)
        self._move_opponent()

        # 3. Move Ball
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # 4. Collisions
        # Top/Bottom Walls
        if self.ball_y <= 0 or self.ball_y >= self.screen_height - self.ball_size:
            self.ball_vy *= -1

        # Paddles
        # Left Paddle
        left_paddle_rect = pygame.Rect(50, self.left_paddle_y, self.paddle_width, self.paddle_height)
        # Right Paddle
        right_paddle_rect = pygame.Rect(self.screen_width - 50 - self.paddle_width, self.right_paddle_y, self.paddle_width, self.paddle_height)
        ball_rect = pygame.Rect(self.ball_x, self.ball_y, self.ball_size, self.ball_size)

        if ball_rect.colliderect(left_paddle_rect):
            self.ball_vx = abs(self.ball_vx) # Bounce right
            # Add some randomness to y velocity to prevent loops
            self.ball_vy += self.np_random.uniform(-1, 1)
            # Clamp ball velocity to prevent unrealistic speeds
            self.ball_vy = np.clip(self.ball_vy, -self.ball_speed_val * 1.5, self.ball_speed_val * 1.5)
            
        if ball_rect.colliderect(right_paddle_rect):
            self.ball_vx = -abs(self.ball_vx) # Bounce left
            self.ball_vy += self.np_random.uniform(-1, 1)
            # Clamp ball velocity to prevent unrealistic speeds
            self.ball_vy = np.clip(self.ball_vy, -self.ball_speed_val * 1.5, self.ball_speed_val * 1.5)

        # 5. Scoring & Rewards
        reward = 0.0
        terminated = False
        
        # REWARD SHAPING: Give intermediate rewards for good positioning
        # This helps the agent learn faster than sparse +1/-1 rewards
        
        # Small reward for staying close to ball's y position (tracking behavior)
        paddle_center = self.right_paddle_y + self.paddle_height / 2
        ball_center = self.ball_y + self.ball_size / 2
        distance_to_ball = abs(paddle_center - ball_center)
        
        # Normalize distance (max is screen_height / 2)
        normalized_distance = distance_to_ball / (self.screen_height / 2)
        
        # Small reward for good positioning (closer = better)
        reward += 0.01 * (1.0 - normalized_distance)
        
        # Reward for successfully hitting the ball
        if ball_rect.colliderect(right_paddle_rect):
            reward += 0.1  # Bonus for successful paddle hit
        
        # Terminal rewards
        if self.ball_x < 0:
            # Ball went past Left Paddle -> Agent wins point
            self.score_right += 1
            reward = 1.0  # Override with terminal reward
            terminated = True
        elif self.ball_x > self.screen_width:
            # Ball went past Right Paddle -> Opponent wins point
            self.score_left += 1
            reward = -1.0  # Override with terminal reward
            terminated = True

        # Render
        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, False, {}

    def _move_opponent(self):
        """
        Logic for the Left Paddle (Opponent).
        Modes: human keyboard, loaded policy, or perfect AI.
        The perfect AI predicts where the ball will be and moves there.
        """
        # Human mode: Use keyboard input stored via set_opponent_action
        if self.human_mode:
            if self.human_action == 1:  # Up
                self.left_paddle_y -= self.paddle_speed
            elif self.human_action == 2:  # Down
                self.left_paddle_y += self.paddle_speed
        elif self.opponent_policy is not None:
            # Predict action using the trained model
            raw_obs = self._get_obs()
            flipped_obs = np.fliplr(raw_obs)
            
            gray = cv2.cvtColor(flipped_obs, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
            
            if not hasattr(self, 'opponent_obs_buffer'):
                self.opponent_obs_buffer = np.zeros((84, 84, 4), dtype=np.uint8)
            
            self.opponent_obs_buffer = np.roll(self.opponent_obs_buffer, -1, axis=-1)
            self.opponent_obs_buffer[:, :, -1] = resized
            
            obs_input = np.expand_dims(self.opponent_obs_buffer, axis=0) 
            action, _ = self.opponent_policy.predict(obs_input, deterministic=True)
            action = action.item()
            
            if action == 1:
                self.left_paddle_y -= self.paddle_speed
            elif action == 2:
                self.left_paddle_y += self.paddle_speed
        else:
            # PERFECT OPPONENT: Predict where ball will intercept left paddle x-position
            # This creates a very challenging opponent that the agent must learn to beat
            target_y = self._predict_ball_intercept_y()
            paddle_center = self.left_paddle_y + self.paddle_height / 2
            
            # Move towards predicted intercept point
            if target_y < paddle_center - 2:  # Small deadzone to prevent jitter
                self.left_paddle_y -= self.paddle_speed
            elif target_y > paddle_center + 2:
                self.left_paddle_y += self.paddle_speed
            
        self.left_paddle_y = np.clip(self.left_paddle_y, 0, self.screen_height - self.paddle_height)

    def _predict_ball_intercept_y(self):
        """
        Predict where the ball will be when it reaches the left paddle's x position.
        This simulates the ball trajectory to give the opponent perfect information.
        """
        # Left paddle x position (where we want to intercept)
        target_x = 50 + self.paddle_width
        
        # If ball is moving away from left paddle, just track current position
        if self.ball_vx > 0:
            return self.ball_y + self.ball_size / 2
        
        # Simulate ball trajectory
        sim_x = self.ball_x
        sim_y = self.ball_y
        sim_vx = self.ball_vx
        sim_vy = self.ball_vy
        
        # Maximum simulation steps to prevent infinite loop
        max_steps = 500
        for _ in range(max_steps):
            sim_x += sim_vx
            sim_y += sim_vy
            
            # Bounce off top/bottom walls
            if sim_y <= 0 or sim_y >= self.screen_height - self.ball_size:
                sim_vy *= -1
                sim_y = np.clip(sim_y, 0, self.screen_height - self.ball_size)
            
            # Check if ball reached target x
            if sim_x <= target_x:
                return sim_y + self.ball_size / 2
        
        # Fallback: return current ball y
        return self.ball_y + self.ball_size / 2

    def set_opponent_action(self, action):
        """Set the opponent's action for human mode (called from play.py)."""
        self.human_action = action
    
    def enable_human_mode(self, enabled=True):
        """Enable/disable human control of the opponent (left paddle)."""
        self.human_mode = enabled

    def update_opponent_model(self, model_path):
        """
        Load a new PPO model for the opponent.
        """
        from stable_baselines3 import PPO
        print(f"Loading opponent model from {model_path}")
        try:
            self.opponent_policy = PPO.load(model_path)
            # Reset buffer on new model load
            self.opponent_obs_buffer = np.zeros((84, 84, 4), dtype=np.uint8)
        except Exception as e:
            print(f"Failed to load opponent model: {e}")

    def _render_frame(self):
        # Fill Background
        self.screen.fill((0, 0, 0)) # Black

        # Draw Paddles (White)
        pygame.draw.rect(self.screen, (255, 255, 255), (50, self.left_paddle_y, self.paddle_width, self.paddle_height))
        pygame.draw.rect(self.screen, (255, 255, 255), (self.screen_width - 50 - self.paddle_width, self.right_paddle_y, self.paddle_width, self.paddle_height))

        # Draw Ball (White)
        pygame.draw.rect(self.screen, (255, 255, 255), (self.ball_x, self.ball_y, self.ball_size, self.ball_size))

        # Draw Middle Line
        pygame.draw.aaline(self.screen, (200, 200, 200), (self.screen_width // 2, 0), (self.screen_width // 2, self.screen_height))
        
        # Draw Scores
        # Note: Drawing text might add complexity to the pixels, but helpful for human watching.
        # Maybe toggle off for pure training if needed, but CNN can learn to ignore it.
        if self.render_mode == "human":
            score_text = self.font.render(f"{self.score_left}  {self.score_right}", True, (255, 255, 255))
            self.screen.blit(score_text, (self.screen_width // 2 - score_text.get_width() // 2, 20))

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def _get_obs(self):
        # Capture the current screen state as an image
        # If in headless mode, we still need to draw to the surface to get pixels
        if self.render_mode != "human":
             self._render_frame()
        
        # Get pixels from surface
        pixel_array = pygame.surfarray.array3d(self.screen)
        # Pygame surface is (W, H, 3), we want (H, W, 3) usually for CV / Gym, or channel first.
        # Gymnasium Box expects (Low, High, Shape).
        # surfarray is transposed (Width, Height, Channels). transpose to (Height, Width, Channels)
        return np.transpose(pixel_array, (1, 0, 2))

    def close(self):
        if self.screen is not None:
             pygame.quit()
