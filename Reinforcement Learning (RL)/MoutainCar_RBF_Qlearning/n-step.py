# Adapt Q-Learning script to use N-step method instead

from __future__ import print_function, division
from builtins import range

import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime

# import from our previous Q-Learning program
import RBF_Qlearning
from RBF_Qlearning import plot_cost_to_go, FeatureTransformer, Model, plot_running_avg


# define our own SGDRegressor
class SGDRegressor:
    def __init__(self, **kwargs):
        self.w = None
        self.lr = 1e-2

    def partial_fit(self, x, y):
        if self.w is None:
            d = x.shape[1]
            self.w = np.random.randn(d) / np.sqrt(d)
        self.w += self.lr*(y - x.dot(self.w)).dot(x)

    def predict(self, x):
        return x.dot(self.w)


# replace the SGDRegressor class in the previous script with ours above
RBF_Qlearning.SGDRegressor = SGDRegressor


# returns a list of states, rewards and the total reward
def play_one(model, eps, gamma, n=5):
    observation = env.reset()
    done = False
    totalreward = 0
    rewards = []
    states = []
    actions = []
    iters = 0

    # multiplier used for n-step calculation
    multiplier = np.array([gamma]*n)**np.arange(n)

    while not done and iters < 10000:
        action = model.sample_action(observation, eps)
        states.append(observation)
        actions.append(action)

        prev_observation = observation
        observation, reward, done, info = env.step(action)

        rewards.append(reward)

        # update the model
        if len(rewards) >= n:
            return_up_to_prediction = multiplier.dot(rewards[-n:])
            g = return_up_to_prediction + (gamma**n)*np.max(model.predict(observation)[0])
            model.update(states[-n], actions[-n], g)

        totalreward += reward
        iters += 1

    # empty the cache
    if n == 1:
        rewards = []
        states = []
        actions = []
    else:
        rewards = rewards[-n+1:]
        states = states[-n+1:]
        actions = actions[-n+1:]

    # check to see if the car has reached the goal of postion >= 0.5
    if observation[0] >= 0.5:
        print("Made It!!")
        while len(rewards) > 0:
            g = multiplier[:len(rewards)].dot(rewards)
            model.update(states[0], actions[0], g)
            rewards.pop(0)
            states.pop(0)
            actions.pop(0)
    else:
        print("Didn't Make it")
        while len(rewards) > 0:
            guess_rewards = rewards + [-1]*(n - len(rewards))
            g = multiplier.dot(guess_rewards)
            model.update(states[0], actions[0], g)
            rewards.pop(0)
            states.pop(0)
            actions.pop(0)

    return totalreward


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    ft = FeatureTransformer(env)
    model = Model(env, ft, "constant")
    gamma = 0.99

    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = "./" + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

    N = 300
    totalrewards = np.empty(N)
    costs = np.empty(N)
    for n in range(N):
        eps = 0.1*(0.97**n)
        totalreward = play_one(model, eps, gamma)
        totalrewards[n] = totalreward
        print("Episode:", n, "Total Reward:", totalreward)
    print("Average reward for the last 100 episodes:", totalrewards[-100:].mean())
    print("Total Steps:", -totalrewards.sum())

    plt.plot(totalrewards)
    plt.title("Total Rewards")
    plt.show()

    plot_running_avg(totalrewards)

    # plot the optimal state-value function
    plot_cost_to_go(env, model)










