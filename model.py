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
        """
        Saves a given state to memory.

        Parameters
        ----------
        state: tuple (float, float, float, float)
            The observation state to save.
        
        Returns
        -------
        none
        """
        INITIAL_WEIGHT = 0.0

        if state not in self.memory:
            self.memory.update({state : {0 : INITIAL_WEIGHT, 1 : INITIAL_WEIGHT}})
    
    def getAction(self, actionSpace, state):
        """
        Gets the action to perform at a state.

        Parameters
        ----------
        actionSpace: action_space
            The action space of the problem.
        state: tuple (float, float, float, float)
            The observation state to find an action for.
        
        Returns
        -------
        int - The action to perform.
        """
        if random() > EPSILON: # Check for exploration
            return max(self.memory[state][0], self.memory[state][1])
        else:
            return actionSpace.sample()