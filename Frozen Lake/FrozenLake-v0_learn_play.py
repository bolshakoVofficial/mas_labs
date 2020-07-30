import gym
import numpy as np
import time
import pickle

env = gym.make('FrozenLake-v0')
env.render()

total_episodes = 20000
max_steps = 100
gamma = 0.9
lr_rate = 0.5

# parameters for epsilon
x = np.linspace(1, total_episodes+1, total_episodes)
epsilon = x ** 0.75 * np.log(x)
divider = np.amax(epsilon)
shift = 0.65
epsilon_norm = epsilon / (5 * divider) + shift

Q = np.zeros((env.observation_space.n, env.action_space.n))


def choose_action(state, epsilon):
    action = 0
    if np.random.uniform(0, 1) < (1 - epsilon):
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action


def choose_action2(state):
    action = 0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action


def choose_action3(state):
    action = np.argmax(Q[state, :])
    return action


def learn(state, state2, reward, action):
    Q[state, action] = Q[state, action] + lr_rate * \
                       (reward + gamma * np.max(Q[state2, :]) - Q[state, action])


# Start
for episode in range(total_episodes):
    state = env.reset()
    t = 0

    while t < max_steps:
        # env.render()

        action = choose_action(state, epsilon_norm[episode])
        # action = choose_action2(state)

        state2, reward, done, info = env.step(action)

        # print("state{} --action{}--> state{}".format(state, action, state2))

        if done and (reward == 0):
            reward = -10
        elif done and (reward == 1):
            reward = 100
        else:
            reward = -1

        learn(state, state2, reward, action)
        state = state2
        t += 1

        if done:
            break

# print(Q)

with open("frozenLake_qTable.pkl", 'wb') as f:
    pickle.dump(Q, f)

'''***************************************************************
    ************************** playing ***************************
    **************************************************************
    **************** comment below if don't need ******************
    **************************************************************'''

with open("frozenLake_qTable.pkl", 'rb') as f:
    Q = pickle.load(f)

win = 0
defeat = 0

# start
for episode in range(1000):

    state = env.reset()
    # print("Episode:", episode, end='')
    t = 0
    while t < 100:
        # env.render()

        action = choose_action3(state)

        state2, reward, done, info = env.step(action)

        if done and (reward == 1):
            # print(" \tWIN ")
            win += 1
        if done and (reward == 0):
            # print(" \tDEFEAT ")
            defeat += 1

        state = state2

        if done:
            break

winrate = win / (win + defeat) * 100
print("Winrate: {}%".format(winrate))
