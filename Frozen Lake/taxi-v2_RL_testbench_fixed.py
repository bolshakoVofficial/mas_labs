import numpy as np
import gym, os
import random, pickle


def choose_action(epsilon):
    if random.uniform(0, 1) < (1 - epsilon):
        action = env.action_space.sample()  # Explore action space
    else:
        action = np.argmax(q_table[state])  # Exploit learned values
    return action


def learn(state, state2, reward, action):
    q_table[state, action] = q_table[state, action] + alpha * \
                             (reward + gamma * np.max(q_table[state2, :]) - q_table[state, action])


env = gym.make("Taxi-v2").env

learning_episodes = 100000
playing_episodes = 300000

epsilon = 0.6
alpha = 0.1
gamma = 0.1
min_spe = 12.52

for epsilon_epoch in range(7):
    print(f"- - - Training: {epsilon_epoch}", end=", ")
    epsilon += 0.05
    gamma = 0.1
    print("(epsilon = {})".format(epsilon))

    for gamma_epoch in range(8):
        print(f"- - Gamma: {gamma_epoch}", end=" ")
        gamma += 0.1
        alpha = 0.1
        print("(gamma = {})".format(gamma))

        for alpha_epoch in range(8):
            print(f"- Alpha: {alpha_epoch}", end=" ")
            alpha += 0.1
            print("(alpha = {})".format(alpha))

            total_epochs, total_penalties, total_reward = 0, 0, 0
            q_table = np.zeros([env.observation_space.n, env.action_space.n])

            # learning
            for i in range(0, learning_episodes):
                state = env.reset()
                done = False

                while not done:
                    action = choose_action(epsilon)
                    next_state, reward, done, info = env.step(action)
                    learn(state, next_state, reward, action)
                    state = next_state
            print("Learned ->", end=" ")

            # playing
            for _ in range(playing_episodes):
                state = env.reset()
                epochs, penalties, reward = 0, 0, 0
                done = False

                while not done:
                    action = np.argmax(q_table[state])
                    state, reward, done, info = env.step(action)

                    if reward == -10:
                        penalties += 1

                    epochs += 1
                    total_reward += reward

                total_penalties += penalties
                total_epochs += epochs

            steps_per_episode = total_epochs / playing_episodes
            reward_per_step = total_reward / total_epochs
            print(f"Steps per ep: {steps_per_episode}, Reward per step: {reward_per_step}")

            if (total_penalties == 0) and (steps_per_episode < min_spe):
                min_spe = steps_per_episode
                with open("taxi_eps{0}_gm{1:.2f}_al{2:.2f}_spe{3}.pkl"
                                  .format(epsilon, gamma, alpha, steps_per_episode), 'wb') as f:
                    pickle.dump(q_table, f)
