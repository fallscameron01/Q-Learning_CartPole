import numpy as np
import gym

from model import Model

if __name__ == '__main__':
    NUM_EPISODES = 1000
    MAX_STEPS = 500

    env = gym.make("CartPole-v1")

    model = Model(env)

    for episode in range(NUM_EPISODES):
        episodeReward = 0
        obsSpace = env.reset()
        state = model.convertState(obsSpace, env, episode)
        model.saveState(state)

        for step in range(MAX_STEPS):
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