"""
Handles running the environment. Uses Model with q-learning to solve the CartPole-v1 environment from gym.
"""

import numpy as np
import gym

from model import Model

if __name__ == '__main__':
    MAX_EPISODES = 1000 # Max number of episodes
    MAX_STEPS = 500 # Max steps per episode
    EPS_TO_RENDER = 500 # Number of episodes until rendering starts

    env = gym.make("CartPole-v1")

    model = Model(env)

    for episode in range(MAX_EPISODES):
        episodeReward = 0
        obsSpace = env.reset()
        state = model.convertState(obsSpace, env, episode)
        model.saveState(state)

        for step in range(MAX_STEPS):
            if episode > EPS_TO_RENDER:
                env.render()

            action = model.getAction(env.action_space, state)

            newObsSpace, reward, done, info = env.step(action)

            episodeReward += reward

            newState = model.convertState(newObsSpace, env, episode)
            model.saveState(newState)

            model.updateState(reward, state, action, newState)

            obsSpace = newObsSpace
            state = newState

            if done:
                print("Episode", episode, "finished after", step, "steps, with reward", episodeReward)
                break

    print(model.getMemory())
    env.close()