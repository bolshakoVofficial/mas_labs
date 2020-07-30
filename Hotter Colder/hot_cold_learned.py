import gym
import numpy as np
import random
import pickle


with open("hc_q_table.pkl", 'rb') as f:
    q_table = pickle.load(f)

env = gym.make("HotterColder-v0")
env.reset()

step_size = 100
epochs = 1000
victories = 0

for e in range(epochs):
    env.reset()
    guess = np.array([np.random.randint(-2000, 2000)])
    done = False

    while not done:
        if not env.action_space.contains(guess):
            guess = np.array([np.random.randint(-2000, 2000)])

        obs_next, reward, done, info = env.step(guess)

        action = np.argmax(q_table[obs_next])
        number = guess

        if action == 0:
            guess = guess - step_size * (1 - reward)
        else:
            guess = guess + step_size * (1 - reward)

        if -2000.0 <= guess[0] >= 2000.0:
            print('Value is out of range!', guess)
            guess = number

        if (guess[0] - info['number']) < 0.0000001:
            print('Victory!', e)
            victories += 1
            done = True
print('Win rate = {}%'.format(100*victories/epochs))
env.close()
