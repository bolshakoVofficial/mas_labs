import numpy as np
import gym, os, pylab
import random, pickle


def choose_action():
    if random.uniform(0, 1) < (1 - epsilon):
        action = env.action_space.sample()  # Explore action space
    else:
        action = np.argmax(q_table[state])  # Exploit learned values
    return action


def learn(state, state2, reward, action):
    q_table[state, action] = q_table[state, action] + alpha * \
                             (reward + gamma * np.max(q_table[state2, :]) - q_table[state, action])


env = gym.make("Taxi-v2").env
q_table = np.zeros([env.observation_space.n, env.action_space.n])
total_episodes = 100000

# Hyperparameters
alpha = 0.9
gamma = 0.5
epsilon = 0.75

all_epochs = []
epoch = 0
all_penalties = []
all_rewards = []

for i in range(0, total_episodes):
    state = env.reset()
    penalties, total_reward, epoch = 0, 0, 0
    done = False

    while not done:
        action = choose_action()
        next_state, reward, done, info = env.step(action)
        learn(state, next_state, reward, action)
        state = next_state

        if reward == -10:
            penalties += 1
        epoch += 1
        total_reward += reward

    all_epochs.append(epoch)
    all_penalties.append(penalties)
    all_rewards.append(total_reward)

    if i % 10000 == 0:
        print(f"Episode: {i}")

with open("taxi_qTable.pkl", 'wb') as f:
    pickle.dump(q_table, f)

print("Training finished.\n")

# plot
'''x = np.linspace(0, total_episodes, total_episodes)
# pylab.plot(x, all_penalties)
# pylab.plot(x, all_rewards)
pylab.plot(x, all_epochs)
pylab.xlabel('Total episodes')
pylab.ylabel('Epochs')
pylab.show()'''
