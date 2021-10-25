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

# actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

VOID = 0
APPLE = 1
BODY = 2
HEAD = 3


def get_object(matrix, target):
    for y in range(grid_size):
        for x in range(grid_size):
            if matrix[y][x] == target:
                return x, y


def make_move(x_snek, y_snek, x_apple, y_apple, matrix):
    if y_snek % 2 == 0:
        if x_snek == grid_size - 1:
            return DOWN
        if x_snek == 0 and y_snek != 0:
            return UP
        return RIGHT
    else:
        if x_snek == 0:
            return UP
        if x_snek == 1 and y_snek != grid_size - 1:
            return DOWN
        return LEFT

env = SnakeEnv(grid_size=[grid_size, grid_size], snake_size=2)
obs = env.reset()  # construct instance of game
done = False
while not done:
    env.render(frame_speed=.01)
    x_snek, y_snek = get_object(obs, HEAD)
    x_apple, y_apple = get_object(obs, APPLE)
    obs, reward, done, info = env.step(make_move(x_snek, y_snek, x_apple, y_apple, obs))  # pass action to step()

env.close()