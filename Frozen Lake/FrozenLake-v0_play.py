import gym
import numpy as np
import time, pickle, os

env = gym.make('FrozenLake-v0')
win = 0
defeat = 0

with open("frozenLake_qTable.pkl", 'rb') as f:
    Q = pickle.load(f)


# print(Q)


def clear():
    os.system('cls')


def choose_action(state):
    action = np.argmax(Q[state, :])
    return action


# start
for episode in range(3):
    state = env.reset()
    # print("Episode:", episode, end='')
    t = 0
    while t < 100:
        env.render()

        action = choose_action(state)

        state2, reward, done, info = env.step(action)

        if done and (reward == 1):
            print(" \tWIN ")
            win += 1
            time.sleep(.8)
        if done and (reward == 0):
            print(" \tDEFEAT ")
            defeat += 1
            time.sleep(.8)

        state = state2
        time.sleep(.1)
        clear()

        if done:
            break

print("Winrate: {}%".format(win / (win + defeat) * 100))
# print('\n\n\n\n\n\n\n\n')
