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

        model.saveState(obsSpace)

        for step in range(MAX_STEPS):
            env.render()

            action = model.getAction(env.action_space, obsSpace)
            newObsSpace, reward, done, info = env.step(action)

            obsSpace = newObsSpace
            model.saveState(obsSpace)

            if done:
                print("Episode", episode, "finished after", step, "steps")
                break

    env.close()