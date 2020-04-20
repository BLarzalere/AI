import gym

env = gym.make('CartPole-v1')
env.reset()

box = env.observation_space
box

env.action_space
env.action_space.n

done = False
count = 0
while not done:
    observation, reward, done, _ = env.step(env.action_space.sample())
    count +=1
print(count)



