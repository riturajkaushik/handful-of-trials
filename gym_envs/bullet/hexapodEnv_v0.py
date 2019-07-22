import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
 
class HexapodEnv(gym.Env):  
    metadata = {'render.modes': ['human']}   
    def __init__(self):
        self.action_space = spaces.Box(low= np.zeros(36), high=np.ones(36))
        self.observation_space = spaces.Box(low= np.array([-20,-20, -1, -1]), high=np.array([20,20, 1, 1])) #(x, y, sin_theta, cos_theta)

    def step(self, action):
        pass
 
    def reset(self):
        pass
 
    def render(self, mode='human', close=False):
        pass