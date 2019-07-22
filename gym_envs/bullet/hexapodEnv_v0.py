import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
# from simu.hexapod_simu import Hexapod_env 
from gym_envs.bullet.simu.hexapod_simu import Hexapod_env, HexaController 

class HexapodEnv(gym.Env): 
    def __init__(self, goal=np.array([-4, 2.3])):
        self.action_space = spaces.Box(low= np.zeros(36), high=np.ones(36))
        self.observation_space = spaces.Box(low= np.array([-20,-20, -1, -1]), high=np.array([20,20, 1, 1])) #(x, y, sin_theta, cos_theta)
        self.ctlr = HexaController()
        self.hexa = Hexapod_env(gui=True)
        self.hexa.setController(self.ctlr)
        self.sim_time = 3.0
        self.state = np.array([0, 0, np.sin(0), np.cos(0)])
        self.goal = goal

    def __get_state(self):
        cm =  self.hexa.getState()[0:2]
        ang = self.hexa.getEulerAngles()[2]
        ang = np.deg2rad(ang)
        return np.array([cm[0], cm[1], np.sin(ang), np.cos(ang)])
    
    def step(self, action):
        self.hexa.run(action, self.sim_time)
        self.state = self.__get_state()
        diff = (self.state[0]-self.goal[0])**2 +  (self.state[1]-self.goal[1])**2 
        rew = np.exp(-0.05*diff)
        return self.__get_state(), rew, False, {}

    def reset(self):
        self.hexa.reset()
        self.state = self.__get_state()
        return self.state
 
    def render(self, mode='human', close=False):
        pass