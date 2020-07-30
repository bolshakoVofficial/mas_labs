import pylab
import numpy as np

total_episodes = 100000

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

pylab.plot(x, epsilon_norm)
pylab.plot(x, alpha_norm)
# pylab.plot(x, gamma_norm)
pylab.xlabel('Total episodes')
pylab.ylabel('Epsilon')
pylab.show()  # show the plot
