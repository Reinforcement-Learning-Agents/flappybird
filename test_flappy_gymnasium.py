import gymnasium as gym
import flappy_bird_gymnasium  # registra gli env

env = gym.make("FlappyBird-v0", render_mode="human")
obs, info = env.reset()

for _ in range(2000):
    obs, r, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()

env.close()
