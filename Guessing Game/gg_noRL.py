import gym
import numpy as np

env = gym.make("GuessingGame-v0")
env.reset()

guess = 0
lower_lim = -1000
upper_lim = 1000


def observation(guess):
    obs, reward, done, info = env.step(np.array([guess]))
    return obs, reward


for i in range(1, 201):
    print("Step {}".format(i))
    obs, reward = observation(guess)
    if obs == 1:
        print("{} is lower than target".format(guess))
        print(reward)
        lower_lim = guess
    if obs == 3:
        print("{} is higher than target".format(guess))
        print(reward)
        upper_lim = guess
    if obs == 2:
        print("{} is the number".format(guess))
        break

    guess = ((lower_lim + upper_lim) / 2)

env.close()
