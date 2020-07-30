import gym
import numpy as np
import random
import pickle


def choose_action(state):
    if random.uniform(0, 1) < (1 - epsilon):
        action = np.random.randint(0, 2, 1)
    else:
        action = np.argmax(q_table[state])
    return action


def learn(state, next_state, reward, action):
    q_table[state, action] = q_table[state, action] + alpha * \
                             (reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action])


env = gym.make("HotterColder-v0")
env.reset()

states = env.observation_space.n
actions = 2
obs_prev = 0
q_table = np.zeros((states, actions))
step_size = 100

alpha = 0.9
gamma = 0.5
epsilon = 0.75

epochs = 1000

for e in range(epochs):
    env.reset()
    guess = np.array([np.random.randint(-2000, 2000)])
    done = False

    while not done:
        if not env.action_space.contains(guess):
            guess = np.array([np.random.randint(-2000, 2000)])

        obs_next, reward, done, info = env.step(guess)
        action = choose_action(obs_next)
        number = guess

        if action == 0:
            guess = guess - step_size * (1 - reward)
        else:
            guess = guess + step_size * (1 - reward)

        if -2000.0 <= guess[0] >= 2000.0:
            guess = number

        learn(obs_prev, obs_next, reward, action)
        obs_prev = obs_next

        if obs_next == 2:
            print('Victory!', e)
            done = True

print(q_table)
with open("hc_q_table.pkl", 'wb') as f:
    pickle.dump(q_table, f)
env.close()
