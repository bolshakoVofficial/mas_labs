import numpy as np
import matplotlib.pyplot as plt

# Make a fake dataset
height = [70, 75, 60]
bars = ('0.5', '0.63', '0.81')
y_pos = np.arange(len(bars))

plt.bar(y_pos, height)
plt.xticks(y_pos, bars)
plt.yticks(range(60, 80, 5))
plt.xlabel('Learning rate')
plt.ylabel('Average win rate, %')
plt.show()
