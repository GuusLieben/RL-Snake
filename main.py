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

log = logging.getLogger("miniproject_snake")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

NOMOVE = -1
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

#env = SnakeEnv(grid_size=[12, 12], snake_size=2, n_snakes=3, n_foods=3)
env = SnakeEnv(grid_size=[9, 9], snake_size=2, n_snakes=2, n_foods=2)
obs = env.reset()  # construct instance of game
done = False
log.info("start game")
for i in range(30):
    if not done:
        env.render()
        #action = [DOWN, DOWN]  # *normal* multi-player snake: all snakes move at the same time and you receive a list of rewards
        action = DOWN  # turn-based multi-player snake: one snake moves, other snakes don't move
        obs, reward, done, info = env.step(action)  # reward is for the snake that has moved
        log.info("reward: %.0f", reward)
env.close()