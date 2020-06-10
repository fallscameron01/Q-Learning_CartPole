# Q-Learning_CartPole
 Solves the Cart Pole problem using q-learning in the Python OpenAI Gym library.

## How it Works
The program uses the Model class to handle the creation of a Q-State Table. The Model class discretizes the values in the observation space to
create states on a discrete range. The Model will update the Q-Values to determine the best actions to take at any given state. The Q-Values
are calculated using the Q-Learning formula for the next optimal state. The parameters alpha, epsilon, and gamma are used to optimize learning.
The parameters alpha and epsilon are changed after a certain number of episodes to change from a random learning model to a reinforcement learning
model. Rewards at each action are passed to the Model for use in the Q-Learning formula.

![Q-Learning Formula](https://wikimedia.org/api/rest_v1/media/math/render/svg/678cb558a9d59c33ef4810c9618baf34a9577686)

## Requirements
Python Packages:

- gym

## Inspiration
Here are some sources of inspiration I used while creating the project:

https://github.com/coledie/q-learning

https://towardsdatascience.com/cartpole-introduction-to-reinforcement-learning-ed0eb5b58288

https://gym.openai.com/evaluations/eval_EIcM1ZBnQW2LBaFN6FY65g/

### Other Helpful Links

https://gym.openai.com/envs/CartPole-v1/

https://en.wikipedia.org/wiki/Q-learning

https://en.wikipedia.org/wiki/Discretization

https://en.wikipedia.org/wiki/Discretization_of_continuous_features
