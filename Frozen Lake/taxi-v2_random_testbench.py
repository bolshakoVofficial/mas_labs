import gym, random

env = gym.make("Taxi-v2").env

episodes = 100
epochs, penalties, reward = 0, 0, 0
total_epochs, total_penalties, total_reward = 0, 0, 0

for _ in range(episodes):
    env.s = random.randint(0, env.observation_space.n)
    epochs, penalties, reward = 0, 0, 0
    done = False

    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1
        total_reward += reward

    total_penalties += penalties
    total_epochs += epochs

print("Average timesteps: {}".format(total_epochs / episodes))
print("Average penalties: {}".format(total_penalties / episodes))
print("Average reward per move: {}".format(total_reward / total_epochs))
