import numpy as np
import gym, os
import random, pickle, pylab


def choose_action(epsilon):
    if random.uniform(0, 1) < (1 - epsilon):
        action = env.action_space.sample()  # Explore action space
    else:
        action = np.argmax(q_table[state])  # Exploit learned values
    return action


def learn(state, state2, reward, action, alpha, gamma):
    q_table[state, action] = q_table[state, action] + alpha * \
                             (reward + gamma * np.max(q_table[state2, :]) - q_table[state, action])


env = gym.make("Taxi-v2").env
q_table = np.zeros([env.observation_space.n, env.action_space.n])
total_episodes = 200000

x = np.linspace(1, total_episodes + 1, total_episodes)

# epsilon
epsilon = x ** 0.8 * np.log(x)
divider = np.amax(epsilon)
shift = 0.8
epsilon_norm = epsilon / (5 * divider) + shift

# alpha
alpha = - (x ** 10)
divider2 = np.min(alpha)
alpha_norm = alpha / - (4 * divider2) + 0.9

# gamma
gamma = - (x ** 10)
divider2 = np.min(gamma)
gamma_norm = gamma / - (10 * divider2) + 0.9

# fixed parameters
epsilon_fixed = 0.9
alpha_fixed = 0.9
gamma_fixed = 0.6

for i in range(0, total_episodes):
    state = env.reset()
    done = False

    while not done:
        action = choose_action(epsilon_norm[i])
        next_state, reward, done, info = env.step(action)
        learn(state, next_state, reward, action, alpha_norm[i], gamma_fixed)
        state = next_state

    if i % 10000 == 0:
        print(f"Episode: {i}")

with open("taxi_qTable.pkl", 'wb') as f:
    pickle.dump(q_table, f)

print("Training finished.\n")
