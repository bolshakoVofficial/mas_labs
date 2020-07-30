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


env = gym.make("GuessingGame-v0")
env.reset()

states = env.observation_space.n
actions = 2
obs_prev = 0
q_table = np.zeros((states, actions))

alpha = 0.9
gamma = 0.5
epsilon = 0.75

total_episodes = 200
epochs = 100000

lower_bound = -1000
upper_bound = lower_bound * -1

for e in range(epochs):
    env.reset()
    lower = lower_bound
    upper = upper_bound

    if (epsilon < 1) and (e % 20 == 0):
        epsilon += (1 - epsilon) / (total_episodes * 25)
        print("epsilon = ", epsilon)

    for i in range(total_episodes):
        guess = (lower + upper) / 2

        obs_next, reward, done, info = env.step(np.array([guess]))

        action = choose_action(obs_next)
        if action == 0:
            lower = guess
        else:
            upper = guess

        learn(obs_prev, obs_next, reward, action)
        obs_prev = obs_next

        if reward == 1:
            print('Victory!')
            print('Episodes', i)
            break

print(q_table)
with open("gg_q_table.pkl", 'wb') as f:
    pickle.dump(q_table, f)
env.close()
