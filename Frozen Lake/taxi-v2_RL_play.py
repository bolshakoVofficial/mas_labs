import pickle, gym, os
import numpy as np
from time import sleep


# works in Windows cmd
def clear():
    os.system('cls')


with open("taxi_qTable.pkl", 'rb') as f:
    q_table = pickle.load(f)

env = gym.make("Taxi-v2").env

total_epochs, total_penalties, total_reward = 0, 0, 0
episodes = 100000

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    done = False

    while not done:
        # env.render()
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1
        total_reward += reward

        # sleep(.1)
        # clear()

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")
print(f"Average reward per move: {total_reward / total_epochs}")
# print("\n\n\n\n\n\n\n")
