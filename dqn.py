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
from stable_baselines import DQN

import warnings
warnings.filterwarnings('ignore')

import sys
args = sys.argv

properties = {
    'train': 'True',
    'steps': '30',
    'learning_rate': '0.001',
    'double_q': 'True',
    'grid_size': '6',
    'exploration_factor': '0.1',
    'exploration_min': '0.02'
}
for arg in args:
    if arg.__contains__('='):
        kv = arg.split('=')
        properties[kv[0]] = kv[1]

print('Using properties:')
for prop_k, prop_v in properties.items():
    print('- ', prop_k, ':', prop_v)

grid_size = int(properties['grid_size'])

NOMOVE = -1
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

env = SnakeEnv(grid_size=[grid_size, grid_size], snake_size=2, n_snakes=1, n_foods=1)

training = bool(properties['train'])
if training:
    model = DQN('MlpPolicy', env,
                learning_rate=float(properties['learning_rate']),
                verbose=1,
                double_q=bool(properties['double_q']),
                tensorboard_log='tensorboard_logs/snake_dqn/',
                exploration_final_eps=float(properties['exploration_min']),
                exploration_fraction=float(properties['exploration_factor']))
    model.learn(total_timesteps=400_000)
    model.save('learned_models/snake_dqn')
else:
    model = DQN.load('learned_models/snake_dqn')

obs = env.reset()
done = False
for i in range(int(properties['steps'])):
    if not done:
        action, state = model.predict(obs)
        obs, reward, done, info = env.step(action)
        if not training:
            env.render()
env.close()