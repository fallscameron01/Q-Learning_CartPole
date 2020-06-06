import numpy as np
import gym

from model import Model

if __name__ == '__main__':
    NUM_EPISODES = 500
    MAX_STEPS = 500

    model = Model()

    env = gym.make("CartPole-v1")

    for episode in range(NUM_EPISODES):
        obsSpace = env.reset()
        state = model.convertState(obsSpace)
        model.saveState(state)

        for step in range(MAX_STEPS):
            env.render()

            action = model.getAction(env.action_space, state)

            newObsSpace, reward, done, info = env.step(action)

            newState = model.convertState(newObsSpace)
            model.saveState(newState)

            obsSpace = newObsSpace
            state = newState

            if done:
                print("Episode", episode, "finished after", step, "steps")
                break

    env.close()