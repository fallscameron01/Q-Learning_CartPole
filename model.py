import numpy as np
import gym

GAMMA = 0.90
APLHA = 0.01

class Model:
    def __init__(self):
        self.memory = np.array({})
    
    def saveState(self, state):
        self.memory = np.append(self.memory, state)
    
    def getAction(self, actionSpace):
        return actionSpace.sample()