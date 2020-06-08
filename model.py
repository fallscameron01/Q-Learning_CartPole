from random import random
from math import radians

import numpy as np
import gym

class Model:
    def __init__(self, env):
        self.memory = {} # {(observation space) : {action : weight}}

        # Create discrete buckets for storing states
        NUM_BUCKETS = [1, 1, 6, 12] # Number of Buckets for each observation value

        CART_VEL_RANGE = .5 # Range of values to create bucket
        POLE_VEL_RANGE = radians(50) # Range of values to create bucket

        BUCKETS_CART_POS = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=NUM_BUCKETS[0])
        BUCKETS_CART_VEL = np.linspace(-CART_VEL_RANGE, CART_VEL_RANGE, num=NUM_BUCKETS[1])
        BUCKETS_POLE_ANG = np.linspace(env.observation_space.low[2], env.observation_space.high[2], num=NUM_BUCKETS[2])
        BUCKETS_POLE_VEL = np.linspace(-POLE_VEL_RANGE, POLE_VEL_RANGE, num=NUM_BUCKETS[3])

        self.buckets = np.array([BUCKETS_CART_POS, BUCKETS_CART_VEL, BUCKETS_POLE_ANG, BUCKETS_POLE_VEL])
        
        self.alpha = .6 # Learning Rate
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

        cartPos = np.digitize(obsSpace[0], self.buckets[0])
        cartVel = np.digitize(obsSpace[0], self.buckets[1])
        poleAng = np.digitize(obsSpace[0], self.buckets[2])
        poleVel = np.digitize(obsSpace[0], self.buckets[3])

        return (cartPos, cartVel, poleAng, poleVel)

    def getMemory(self):
        """
        Returns the saved memory (the q table).

        Returns
        -------
        dictionary: The saved q table.
        """
        return self.memory

    def _updateParams(self, episode):
        if(episode > 300):
            self.alpha = .1
            self.epsilon = .1
