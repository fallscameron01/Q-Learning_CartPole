"""
Model is used to handle q-learning for the environment.
"""

from random import random
from math import radians

import numpy as np
import gym

class Model:
    """
    Model of q-learning for the environment.

    Parameters
    ----------
    env: environment
        The environment being used.
    """
    def __init__(self, env):
        self.memory = {} # {(observation space) : {action : weight}}

        # Create discrete buckets for storing states
        self.NUM_BUCKETS = [1, 1, 6, 12] # Number of Buckets for each observation value

        self.UPPER_BOUNDS = [env.observation_space.high[0], 0.5, env.observation_space.high[2], radians(50)]
        self.LOWER_BOUNDS = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -radians(50)]

        self.alpha = .5 # Learning Rate
        self.gamma = 1 # Discount Factor
        self.epsilon = 1 # Exploration Rate
    
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

    def updateState(self, reward, state, action, nextState):
        """
        Updates the state saved in memory using the Q-Learning formula.

        Parameters
        ----------
        reward: int
            The reward of the action at the current state.
        state: tuple (float, float, float, float)
            The state the action was performed at.
        action: int
            The action taken.
        nextState: tuple (float, float, float, float)
            The next state of the environment.

        Returns
        -------
        none
        """
        self.memory[state][action] += self.alpha * (reward + self.gamma * max(self.memory[nextState].values()) - self.memory[state][action])
    
    def getAction(self, actionSpace, state):
        """
        Gets the action to perform at a state.

        Parameters
        ----------
        actionSpace: ndarray
            The action space of the problem.
        state: tuple (float, float, float, float)
            The observation state to find an action for.
        
        Returns
        -------
        int: The action to perform.
        """
        if random() > self.epsilon: # Check for exploration
            return max(self.memory[state], key=self.memory[state].get)
        else:
            return actionSpace.sample()

    def convertState(self, obsSpace, env, episode):
        """
        Converts an observation state into a tuple state that can be stored.

        Parameters
        ----------
        obsSpace: ndarray
            The observation state to convert.

        Returns
        -------
        tuple (float, float, float, float): The state as a tuple.
        """
        self._updateParams(episode)

        state = np.arange(len(self.NUM_BUCKETS))

        for i in range(len(self.NUM_BUCKETS)):
            ratio = self._getRatio(i, obsSpace[i])
            pos = int(round(ratio * (self.NUM_BUCKETS[i]-1)))

            # Fit to range
            pos = max(0, pos)
            pos = min(self.NUM_BUCKETS[i]-1, pos)

            state[i] = pos
        
        return tuple(state)
        

    def getMemory(self):
        """
        Returns the saved memory (the q table).

        Returns
        -------
        dictionary: The saved q table.
        """
        return self.memory

    def _updateParams(self, episode):
        """
        Updates the parameters for the q-learning function after a certain number of episodes.

        Parameters
        ----------
        episode: int
            The current episode.

        Returns
        -------
        none
        """
        CHANGE_AFTER = 225

        if(episode > CHANGE_AFTER):
            self.alpha = .1
            self.epsilon = .1

    def _getRatio(self, index, data):
        """
        Returns the ratio for a observation. The ratio is the amount into the discrete range.

        Parameters
        ----------
        index: int
            The index of the data in the observation space.
        data: float
            The data from the current observation space.

        Returns
        -------
        float: The ratio of the data in the discrete range.
        """
        return (data + abs(self.LOWER_BOUNDS[index])) / (self.UPPER_BOUNDS[index] - self.LOWER_BOUNDS[index])