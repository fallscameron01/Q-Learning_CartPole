import numpy as np
import gym


if __name__ == '__main__':
    NUM_EPISODES = 500
    MAX_STEPS = 100

    env = gym.make("CartPole-v1")

    for episode in range(NUM_EPISODES):
        obs_space = env.reset()

        for step in range(MAX_STEPS):
            env.render()

            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            if done:
                break

    env.close()