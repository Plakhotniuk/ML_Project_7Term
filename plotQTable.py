import matplotlib.pyplot as plt
import numpy as np



def get_q_color(value, vals):
    if value == max(vals):
        return "green", 1.0
    else:
        return "red", 0.3


fig, axs = plt.subplots(3, 1, figsize=(12, 9), tight_layout=True)

q_table = np.load(f"data/qtable.npy")
pos_space = np.linspace(-1.2, 0.6, 40)
vel_space = np.linspace(-0.07, 0.07, 40)


for x, x_vals in enumerate(q_table):
    for y, y_vals in enumerate(x_vals):
        axs[0].scatter(pos_space[x], vel_space[y], c=get_q_color(y_vals[0], y_vals)[0], marker="o", alpha=get_q_color(y_vals[0], y_vals)[1])
        axs[1].scatter(pos_space[x],  vel_space[y], c=get_q_color(y_vals[1], y_vals)[0], marker="o", alpha=get_q_color(y_vals[1], y_vals)[1])
        axs[2].scatter(pos_space[x],  vel_space[y], c=get_q_color(y_vals[2], y_vals)[0], marker="o", alpha=get_q_color(y_vals[2], y_vals)[1])

        axs[0].set_title('Action 0 (drive left)')
        axs[1].set_title('Action 1 (stay neutral)')
        axs[2].set_title('Action 2 (drive right)')

        axs[0].set_ylabel("velocity")
        axs[1].set_ylabel("velocity")
        axs[2].set_ylabel("velocity")
        axs[2].set_xlabel("coordinate")

plt.savefig('plots/qtable.png')
plt.show()
