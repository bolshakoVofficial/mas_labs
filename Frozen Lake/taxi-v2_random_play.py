import gym, os
from IPython.display import clear_output
from time import sleep

env = gym.make("Taxi-v2").env
env.s = 328  # set environment to needed state

episodes = 100
epochs, penalties, reward = 0, 0, 0
frames = []  # for animation
done = False


def clear():
    os.system('cls')


def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.05)
        clear()


while not done:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

    if reward == -10:
        penalties += 1

    # Put each rendered frame into dict for animation
    frames.append({
        'frame': env.render(mode='ansi'),
        'state': state,
        'action': action,
        'reward': reward
    }
    )

    epochs += 1

print_frames(frames)
print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))
print('\n\n\n\n\n\n\n\n\n\n')
