from stable_baselines.common.schedules import LinearSchedule
from gym_snake.envs.snake_env import SnakeEnv
from stable_baselines.ppo2 import PPO2
import tensorflow as tf
import time
import sys


# -------- <PROPERTIES> -------- #
properties = {
    'train': 'True',
    'steps': '200',
    'grid_size': '6',
    'n_steps': '128',
    'load_model': None,
    'apples': 1
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
env = SnakeEnv(grid_size=[grid_size, grid_size], snake_size=2, n_snakes=1, n_foods=int(properties['apples']))

training = properties['train'] == 'True'

if not training and properties['load_model'] is None:
    raise 'Training mode is disabled but no existing model was defined'

if training:
    model = PPO2('MlpPolicy', env,
                 verbose=1,
                 tensorboard_log="tensorboard_logs/snake_dqn/",
                 n_steps=int(properties['n_steps'])
    )
    model.learn(1000000)
    model.save('learned_models/snake_ppo2')
else:
    model = PPO2.load('learned_models/' + properties['load_model'])
# -------- </MODEL TRAINING> -------- #


# -------- <DEMONSTRATION> -------- #
obs = env.reset()
done = False
for i in range(int(properties['steps'])):
    if not done:
        action, state = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
env.close()
# -------- <!DEMONSTRATION> -------- #
