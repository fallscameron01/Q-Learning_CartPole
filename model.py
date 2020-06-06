from random import random

import numpy as np
import gym

APLHA = 0.1 # Learning Rate
GAMMA = 0.90 # Discount Factor
EPSILON = 0.1 # Exploration Rate

class Model:
    def __init__(self):
        self.memory = {} # {(observation space) : {action : reward}}
    
    def saveState(self, state):
        INITIAL_WEIGHT = 0.0

        if state not in self.memory:
            self.memory.update({state : {0 : INITIAL_WEIGHT, 1 : INITIAL_WEIGHT}})
    
    def getAction(self, actionSpace, state):
        if random() > EPSILON:
            return max(self.memory[state][0], self.memory[state][1])
        else:
            return actionSpace.sample()