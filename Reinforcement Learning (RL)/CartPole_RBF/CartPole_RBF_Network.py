# Q-Learning using a Radial Basis Function (RBF) network / kernel
from __future__ import print_function, division
from builtins import range

import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler


class SGDRegressor:
    def __init__(self, d):
        self.w = np.random.randn(d) / np.sqrt(d)
        self.lr = 0.1

    # perform one step of gradient descent
    def partial_fit(self, X, Y):
        self.w += self.lr*(Y - X.dot(self.w)).dot(X)

    def predict(self, X):
        return X.dot(self.w)
    # dot product of the input with the weights


class FeatureTransformer:
    def __init__(self, env):
        # gathers 20,000 random samples uniformly from the state space
        observation_examples = np.random.random((20000, 4))*2 - 1

        # standardize the data to have mean of zero and variance of 1
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        # Used to convert a state to a featurized  representation
        # We use RBF kernels with different variances to cover different parts of the space
        # The number of components we pass into the constructor equates to the # of exemplars / centers
        n_components = 1000
        featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=0.05, n_components=n_components)),
            ("rbf2", RBFSampler(gamma=1.0, n_components=n_components)),
            ("rbf3", RBFSampler(gamma=0.5, n_components=n_components)),
            ("rbf4", RBFSampler(gamma=0.1, n_components=n_components))
            ])

        # fit samples to the scaled data
        feature_examples = featurizer.fit_transform(scaler.transform(observation_examples))

        # create instance variables
        self.dimensions = feature_examples.shape[1]
        # want to keep track of the dimensions for later initializing our SGDRegressor
        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observations):
        scaled = self.scaler.transform(observations)
        return self.featurizer.transform(scaled)

# *******************************************
# Note: Sci-Kit Learn inputs must be 2 dimensional, so wrapping 1-D elements/variables in a list makes
# them 2 dimensional with shape (1,1)
# *******************************************


# creates a SGDRegressor for each action; i.e. creates a collection of models
class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer

        for i in range(env.action_space.n):
            model = SGDRegressor(feature_transformer.dimensions)
            self.models.append(model)

    # transforms state into a feature vector & makes a prediction of values - one for each action
    def predict(self, s):
        x = self.feature_transformer.transform(np.atleast_2d(s))
        result = np.stack([m.predict(x) for m in self.models]).T
        return result

    def update(self, s, a, G):
        x = self.feature_transformer.transform(np.atleast_2d(s))
        # calling the partial fit method, but only for the model that corresponds to the action we took
        self.models[a].partial_fit(x, [G])

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))


# returns a list of states, rewards and the total reward from playing one episode
def play_one(env, model, eps, gamma):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0

    while not done and iters < 2000:
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, info = env.step(action)

        # give the agent a large negative reward if the pole falls down
        if done:
            reward = -200

        # update the model
        next = model.predict(observation)
        assert(next.shape == (1, env.action_space.n))
        G = reward + gamma * np.max(next)
        model.update(prev_observation, action, G)

        if reward == 1: # prevent overwriting the -200 negative behavior reward if assigned
            totalreward += reward
        iters += 1

    return totalreward


def plot_running_avg(totalrewards):
    n = len(totalrewards)
    running_avg = np.empty(n)
    for t in range(n):
        running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()


def main():
    env = gym.make('CartPole-v1')
    ft = FeatureTransformer(env)
    model = Model(env, ft)
    gamma = 0.99

    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

    N = 500
    totalrewards = np.empty(N)
    for n in range(N):
        eps = 1.0 / np.sqrt(n+1)
        totalreward = play_one(env, model, eps, gamma)
        totalrewards[n] = totalreward
        if n % 100 == 0:
            print("Episode:", n, "Epsilon:", eps, "Total Reward:", totalreward)

    print("Average reward for last 100 episodes:", totalrewards[-100:].mean())
    print("Total Steps:", totalrewards.sum())

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)


if __name__ == '__main__':
    main()