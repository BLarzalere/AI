# Q-Learning using a Radial Basis Function (RBF) network / kernel
from __future__ import print_function, division
from builtins import range

import gym
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gym import wrappers
from datetime import datetime
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor

# SGDRegressor defaults:
# loss='squared_loss', penalty='l2', alpha=0.0001,
# l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=True,
# verbose=0, epsilon=0.1, random_state=None, learning_rate='invscaling',
# eta0=0.01, power_t=0.25, warm_start=False, average=False
# Additional notes: penalty = regularization type, so l2 in this case. invscaling denotes using an
# inverse scale of the learning rate so that it decreases by 1/t


class FeatureTransformer:
    def __init__(self, env, n_components=500):
        # gathers 10,000 samples from the state space
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])

        # standardize the data to have mean of zero and variance of 1
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        # Used to convert a state to a featurized  representation
        # We use RBF kernels with different variances to cover different parts of the space
        # The number of components we pass into the constructor equates to the # of exemplars / centers
        featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=n_components))
            ])

        # fit samples to the scaled data
        example_features = featurizer.fit_transform(scaler.transform(observation_examples))

        # create instance variables
        self.dimensions = example_features.shape[1]
        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observations):
        scaled = self.scaler.transform(observations)
        return self.featurizer.transform(scaled)

# *******************************************
# Note: Sci-Kit Learn inputs must be 2 dimensional, so wrapping 1-D elements/variables in a list makes
# them 2 dimensional with shape (1,1)
# *******************************************


# creates a SGDRegressor for each action; i.e. a collection of models
class Model:
    def __init__(self, env, feature_transformer, learning_rate):
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer

        for i in range(env.action_space.n):
            model = SGDRegressor(learning_rate=learning_rate)
            # perform one epoch of stochastic gradient decent on given samples
            # partial_fit(X, y) i.e. input, target
            # here, within the constructor initial input is a reset environment and initial target is zero
            model.partial_fit(feature_transformer.transform([env.reset()]), [0])
            self.models.append(model)

    # transforms state into a feature vector & makes a prediction of values - one for each action
    def predict(self, s):
        x = self.feature_transformer.transform([s])
        result = np.stack([m.predict(x) for m in self.models]).T
        assert(len(result.shape) == 2)
        return result

    def update(self, s, a, G):
        x = self.feature_transformer.transform([s])
        assert(len(x.shape) == 2)
        # calling partial fit, but only for the model that corresponds to the action we took
        self.models[a].partial_fit(x, [G])

    def sample_action(self, s, eps):
        # Technically, we don't need to do epsilon-greedy
        # because SGDRegressor predicts 0 for all states
        # until they are updated. This works as the
        # "Optimistic Initial Values" method, since all
        # the rewards for Mountain Car are -1.
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))


# returns a list of states and rewards and the total reward from playing one episode
def play_one(model, env, eps, gamma):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0

    while not done and iters < 10000:
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, info = env.step(action)

        # update the model
        next = model.predict(observation)
        G = reward + gamma*np.max(next[0])
        model.update(prev_observation, action, G)

        totalreward += reward
        iters += 1

    return totalreward


# The cost-to-go function is the negative of the optimal value function
def plot_cost_to_go(env, estimator, num_tiles=20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)
    # both X and Y will be of shape (num_tiles, num_tiles)
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))
    # Z will also be of shape (num_tiles, num_tiles)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, vmin=1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost-To-Go = -V(s)')
    ax.set_title('Cost-To-Go Function')
    fig.colorbar(surf)
    plt.show()


def plot_running_avg(totalrewards):
    n = len(totalrewards)
    running_avg = np.empty(n)
    for t in range(n):
        running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()


def main():
    env = gym.make('MountainCar-v0')
    ft = FeatureTransformer(env)
    model = Model(env, ft, "constant")
    gamma = 0.99

    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

    N = 300
    totalrewards = np.empty(N)
    for n in range(N):
        eps = 0.1*(0.97**n)
        if n == 199:
            print("eps:", eps)
        totalreward = play_one(model, env, eps, gamma)
        totalrewards[n] = totalreward
        if (n + 1) % 100 == 0:
            print("Episode:", n, "Total Reward:", totalreward)

    print("Average reward for last 100 episodes:", totalrewards[-100:].mean())
    print("Total Steps:", -totalrewards.sum())

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)

    # plot the optimal state-value function
    plot_cost_to_go(env, model)


if __name__ == '__main__':
    main()














