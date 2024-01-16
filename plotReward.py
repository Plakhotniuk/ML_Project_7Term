import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

reward = np.loadtxt('data/data_reward_shaping.csv', delimiter=",")
episode = np.arange(len(reward))
plt.plot(episode, reward)
plt.xlabel('episode')
plt.ylabel('reward')
plt.title('Q-learning')
plt.grid()
plt.savefig('plots/rewardShapingEpisode25000.png')
plt.show()
