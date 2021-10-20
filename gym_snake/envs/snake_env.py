import os, subprocess, time, signal
import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from gym_snake.envs.snake import Controller
import logging

try:
    import matplotlib.pyplot as plt
    import matplotlib
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: see matplotlib documentation for installation https://matplotlib.org/faq/installing_faq.html#installation".format(e))

log = logging.getLogger("miniproject_snake")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

'''
Use the coordinate-based environment: self.coordinate_based = False
Use the pixel-based environment: self.coordinate_based = False

when snake leaves world, next world-tick still contains tail without head

'''

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    # Erco
    NOMOVE = -1
    SPACE = 0
    FOOD = 1
    BODY = 2
    HEAD_BASE = 3

    def __init__(self, grid_size=[15,15], unit_size=10, unit_gap=1, snake_size=3, n_snakes=1, n_foods=1, random_init=True):
        self.grid_size = grid_size
        self.unit_size = unit_size
        self.unit_gap = unit_gap
        self.snake_size = snake_size
        self.n_snakes = n_snakes
        self.coordinate_based = True
        self.n_foods = n_foods
        self.viewer = None
        self.action_space = spaces.Discrete(4)
        if self.coordinate_based:
            if self.n_snakes == 1:
                self.observation_space = spaces.Box(low=0, high=self.HEAD_BASE+self.n_snakes-1, shape=(self.grid_size[1], self.grid_size[0]), dtype=np.uint8)  # first y-axis then x-axis for matrix to resemble drawing; '+1 to store current snake'
            else:
                self.observation_space = spaces.Box(low=0, high=self.HEAD_BASE+self.n_snakes-1, shape=(self.grid_size[1] + 1, self.grid_size[0]), dtype=np.uint8)  # first y-axis then x-axis for matrix to resemble drawing; '+1 to store current snake'
        else:  # pixel-based
            if self.n_snakes == 1:
                self.observation_space = spaces.Box(low=0, high=255, shape=(self.grid_size[1] * self.unit_size, self.grid_size[0] * self.unit_size, 3), dtype=np.uint8)
            else:
                self.observation_space = spaces.Box(low=0, high=255, shape=(self.grid_size[1] * self.unit_size + 1, self.grid_size[0] * self.unit_size, 3), dtype=np.uint8) 
        self.random_init = random_init

    # transform from pixel-based to coordinate-based and add current snake
    # note that 'current snake' is the snake that will do the *next* action
    def to_coord(self, state, done):
        state = state[0::self.unit_size]
        state = [ state[i][0::self.unit_size] for i in range(len(state)) ]
        coordstate = np.zeros(self.grid_size)
        for i in range(len(state)):
            for j in range(len(state[0])):
                if state[i][j][1] == 255:  # green in RGB [0,255,0]
                    coordstate[i][j] = self.SPACE
                elif state[i][j][0] == 1:  # black in RGB [1,0,0]
                    coordstate[i][j] = self.BODY
                elif state[i][j][2] == 255:  # blue in RGB [0,0,255]
                    coordstate[i][j] = self.FOOD
                elif state[i][j][0] == 255:  # red in RGB [255,self.controller.HEAD_COLOR_DIFF*i,0]
                    head_count = state[i][j][1] / self.controller.HEAD_COLOR_DIFF  # every head gets a different number
                    coordstate[i][j] = self.HEAD_BASE + head_count
                else:
                    assert True, "unexpected state"
        last_row = np.zeros(self.grid_size[0])
        if not done:  # if done, current snake is set to 0
            last_row[0] = self.current_snake
        if self.n_snakes > 1:
            coordstate = np.vstack([coordstate, last_row])
        ###print(self.time_step); print(coordstate)
        ###if self.time_step % 1000 == 0:
        ###    print(self.time_step)
        return coordstate

    def step(self, action):
        # the action passed to step if for the "current" snake; change it to a list with NOMOVE for the other snakes
        if self.n_snakes > 1:
            actions = np.ones(self.n_snakes) * self.NOMOVE
            actions[self.current_snake - self.HEAD_BASE] = action
            action = actions.tolist()
            ###print('self.current_snake {}, action {}'.format(self.current_snake, action))
        self.last_obs, rewards, done, info = self.controller.step(action)
        coord_obs = self.to_coord(self.last_obs, done)
        if self.n_snakes > 1:
            # next *alive* snake gets the turn
            for i in range(self.n_snakes):
                self.current_snake += 1  # change current snake *after* taking action
                temp = self.current_snake - self.HEAD_BASE
                temp %= self.n_snakes
                self.current_snake = temp + self.HEAD_BASE
                if self.current_snake in coord_obs:
                    break
            ###print('next snake player {}'.format(self.current_snake))
        self.time_step += 1
        if self.coordinate_based:
            return coord_obs, rewards, done, info
        else:  # pixel-based
            return self.last_obs, rewards, done, info

    def reset(self):
        self.time_step = 0
        self.controller = Controller(self.grid_size, self.unit_size, self.unit_gap, self.snake_size, self.n_snakes, self.n_foods, random_init=self.random_init)
        self.last_obs = self.controller.grid.grid.copy()
        self.current_snake = self.HEAD_BASE  # turn-based multi-player snake
        coord_obs = self.to_coord(self.last_obs, False)
        if self.coordinate_based:        
            return coord_obs
        else:  # pixel-based
            return self.last_obs

    def render(self, mode='human', close=False, frame_speed=.1):
        if self.viewer is None:
            self.fig = plt.figure()
            self.viewer = self.fig.add_subplot(111)
            plt.ion()
            self.fig.show()
        else:
            self.viewer.clear()
            self.viewer.imshow(self.last_obs)
            plt.pause(frame_speed)
        self.fig.canvas.draw()

    def seed(self, x):
        pass
