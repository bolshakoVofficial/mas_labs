import gym
import numpy as np
import random
import pickle


with open("gg_q_table.pkl", 'rb') as f:
    q_table = pickle.load(f)

env = gym.make("GuessingGame-v0")
env.reset()

total_episodes = 200
lower_bound = -1000
upper_bound = lower_bound * -1
lower = lower_bound
upper = upper_bound

victories = 0
epochs = 100

for e in range(epochs):
    env.reset()
    lower = lower_bound
    upper = upper_bound

    for i in range(total_episodes):
        guess = (lower + upper) / 2

        obs_next, reward, done, info = env.step(np.array([guess]))

        action = np.argmax(q_table[obs_next])
        if action == 0:
            lower = guess
        else:
            upper = guess

        obs_prev = obs_next

        if reward == 1:
            print('Victory! Episodes', i)
            print(obs_next)
            print(guess, info['number'])
            victories += 1
            break
print('Win rate: {}%'.format(100*victories/epochs))
env.close()
