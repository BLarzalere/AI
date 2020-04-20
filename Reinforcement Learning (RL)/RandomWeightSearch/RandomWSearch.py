from __future__ import print_function, division
from builtins import range

import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt


def get_action(s, w):
    return 1 if s.dot(w) > 0 else 0


def play_one_episode(env, params):
    observation = env.reset()
    done = False
    t = 0

    while not done and t < 10000:
        # render method will open animation window of the execution run
        # - very cool, but slows code execution
        # env.render()
        t += 1
        action = get_action(observation, params)
        observation, reward, done, info = env.step(action)
        if done:
            break
    return t


def play_multiple_episodes(env, T, params):
    episode_length = np.empty(T)

    for i in range(T):
        episode_length[i] = play_one_episode(env, params)

    avg_length = episode_length.mean()
    print("Average episode length:", avg_length)

    return avg_length


def random_search(env):
    episode_lengths = []
    best = 0
    params = None

    for p in range(100):
        new_params = np.random.random(4)*2 - 1
        avg_length = play_multiple_episodes(env, 100, new_params)
        # here T = 100 so we test each parameter vector 100 times
        episode_lengths.append(avg_length)

        if avg_length > best:
            params = new_params
            best = avg_length

    return episode_lengths, params


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    episode_lengths, params = random_search(env)
    plt.plot(episode_lengths)
    plt.show()

    # play final set of episodes
    env = wrappers.Monitor(env, '/Users/brent/Sandbox/RL/RandomWeightSearch')
    print("*** Final run using the final weight parameters ***")
    play_multiple_episodes(env, 100, params)



