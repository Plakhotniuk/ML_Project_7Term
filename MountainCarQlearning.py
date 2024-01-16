import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def run(episodes, is_training=True, render=False, reward_shaping=False):

    env = gym.make('MountainCar-v0', render_mode='human' if render else None)

    # Divide position and velocity into segments
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 40)    # Between -1.2 and 0.6
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 40)    # Between -0.07 and 0.07

    if is_training:
        qtable = np.random.uniform(0, 1, (len(pos_space), len(vel_space), env.action_space.n))  # init a 20x20x3 array
    else:
        qtable = np.load('data/qtable.npy')
    learning_rate_a = 0.1  # alpha or learning rate
    discount_factor_g = 0.95  # gamma or discount factor.

    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 2/episodes  # epsilon decay rate
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]      # Starting position, starting velocity always 0
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)

        terminated = False          # True when reached goal

        rewards = 0
        cnt = 0
        while not terminated and cnt < 1000:
            cnt += 1
            if is_training and rng.random() < epsilon:
                # Choose random action (0=drive left, 1=stay neutral, 2=drive right)
                action = env.action_space.sample()
            else:
                # Choose the best action
                action = np.argmax(qtable[state_p, state_v, :])

            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)

            if reward_shaping:
                reward += 300 * (discount_factor_g * abs(new_state_v) - abs(state_v))

            if is_training:
                qtable[state_p, state_v, action] = qtable[state_p, state_v, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(qtable[new_state_p, new_state_v, :]) - qtable[state_p, state_v, action]
                )

            state_p = new_state_p
            state_v = new_state_v

            rewards += reward

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        rewards_per_episode[i] = rewards

        print('Episode: ', i, ' Reward: ', rewards, ' Terminated: ', terminated)

    env.close()

    # Save Q table to file
    if is_training:
        np.save(f"data/qtable{episodes}.npy", qtable)

    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(mean_rewards)
    plt.grid()

    if is_training:
        if reward_shaping:
            plt.savefig(f'plots/mountain_car_reward_shaping{episodes}.png')
            mean_rewards.tofile('data/data_reward_shaping{episodes}.csv', sep=',')
        else:
            plt.savefig(f'plots/mountain_car{episodes}.png')
            mean_rewards.tofile(f'data/data{episodes}.csv', sep=',')


if __name__ == '__main__':
    # run(5000, is_training=True, render=False, reward_shaping=False)

    run(10, is_training=False, render=True)


