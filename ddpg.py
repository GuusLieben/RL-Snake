from gym_snake.envs.snake_env import SnakeEnv
from stable_baselines.ddpg import DDPG
import tensorflow as tf
import time
import sys


# -------- <PROPERTIES> -------- #
properties = {
    'train': 'True',
    'steps': '200',
    'grid_size': '6',
}

for arg in sys.argv:
    if arg.__contains__('='):
        kv = arg.split('=')
        properties[kv[0]] = kv[1]
        
print('Using properties:')
for prop_k, prop_v in properties.items():
    print('- ', prop_k, ':', prop_v)
# -------- </PROPERTIES> -------- #


# -------- <MODEL TRAINING> -------- #
grid_size = int(properties['grid_size'])
env = SnakeEnv(grid_size=[grid_size, grid_size], snake_size=2, n_snakes=1, n_foods=1)

if properties['train'] == 'True':
    model = DDPG('MlpPolicy', env, verbose=1, tensorboard_log="tensorboard_logs/snake_dqn/")
    model.learn(1000000)
    model.save('learned_models/snake_ddpg')
else:
    model = DQN.load('learned_models/snake_ddpg')
# -------- </MODEL TRAINING> -------- #


# -------- <DEMONSTRATION> -------- #
obs = env.reset()
done = False
for i in range(int(properties['steps'])):
    if not done:
        action, state = model.predict(obs)
        obs, reward, done, info = env.step(action)
        if not training:
            env.render()
env.close()
# -------- <!DEMONSTRATION> -------- #