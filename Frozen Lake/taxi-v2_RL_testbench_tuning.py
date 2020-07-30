import numpy as np
import gym, os
import random, pickle


def choose_action(epsilon):
    if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()  # Explore action space
    else:
        action = np.argmax(q_table[state])  # Exploit learned values
    return action


def choose_action2(epsilon):
    if random.uniform(0, 1) < (1 - epsilon):
        action = env.action_space.sample()  # Explore action space
    else:
        action = np.argmax(q_table[state])  # Exploit learned values
    return action


def learn(state, state2, reward, action):
    q_table[state, action] = q_table[state, action] + alpha * \
                             (reward + gamma * np.max(q_table[state2, :]) - q_table[state, action])


def learn2(state, state2, reward, action):
    q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * \
                             (reward + gamma * np.max(q_table[state2]))


env = gym.make("Taxi-v2").env

learning_episodes = 100000
playing_episodes = 10000
x = np.linspace(1, learning_episodes + 1, learning_episodes)
epsilon_distr = x ** 0.6 * np.log(x)
divider = np.amax(epsilon_distr)
shift = 0.79
div_coef = 4
epsilon_norm = epsilon_distr / (div_coef * divider) + shift

epsilon = 0.1
alpha = 0.44
gamma = 0.85
min_spe = 12.52

for sh_div_coef in range(15):
    print(f"- - - Training: {sh_div_coef}", end=", ")
    shift += 0.01
    div_coef += 1
    gamma = 0.85
    epsilon_norm = epsilon_distr / (div_coef * divider) + shift
    print("Epsilon min: {0:.4f}, max: {1:.4f}".format(np.min(epsilon_norm), np.max(epsilon_norm)))

    for gamma_epoch in range(10):
        print(f"- - Gamma: {gamma_epoch}", end=" ")
        gamma += 0.01
        alpha = 0.44
        print("(gamma = {})".format(gamma))

        for alpha_epoch in range(30):
            print(f"- Alpha: {alpha_epoch}", end=" ")
            alpha += 0.01
            print("(alpha = {})".format(alpha))

            total_epochs, total_penalties, total_reward = 0, 0, 0
            q_table = np.zeros([env.observation_space.n, env.action_space.n])

            # learning
            for i in range(0, learning_episodes):
                state = env.reset()
                done = False

                while not done:
                    # action = choose_action(epsilon)
                    action = choose_action2(epsilon_norm[i])
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
                with open("taxi_sdc{0}_gm{1:.2f}_al{2:.2f}.pkl"
                                  .format(sh_div_coef, gamma, alpha), 'wb') as f:
                    pickle.dump(q_table, f)
