# Example agent for *turn-based* multi-player snake
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import sys
import logging
import gym
#import gym_snake  # don't use the registered snake
from gym_snake.envs.snake_env import SnakeEnv

grid_size = 12

NOMOVE = -1
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

env = SnakeEnv(grid_size=[grid_size, grid_size], snake_size=2, n_snakes=1, n_foods=1)
obs = env.reset()  # construct instance of game
done = False
for i in range(30):
    if not done:
        env.render(frame_speed=.05)
        #action = [DOWN, DOWN]  # *normal* multi-player snake: all snakes move at the same time and you receive a list of rewards
        action = DOWN  # turn-based multi-player snake: one snake moves, other snakes don't move
        obs, reward, done, info = env.step(action)  # reward is for the snake that has moved

env.close()