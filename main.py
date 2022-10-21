import random
from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam


def deep_q_learning(env, n_episodes=50, eps=0.1, batch_size=4, gamma=0.8):
    memory = deque(maxlen=2048)
    print(env.observation_space.shape)
    # compile model
    # Convolutions on the frames on the screen

    # q_network = Sequential()
    # q_network.add(Dense(input_shape=env.observation_space.shape, units=8, activation='relu'))
    # q_network.add(Dense(8, activation='relu'))
    # q_network.add(Dropout(0.2))
    # q_network.add(Dense(env.action_space.n, activation='linear'))
    # q_network.compile(loss='mse', optimizer=Adam(learning_rate=0.01))

    # next_q_network = Sequential()
    # next_q_network.add(Dense(input_shape=env.observation_space.shape, units=8, activation='relu'))
    # next_q_network.add(Dense(8, activation='relu'))
    # next_q_network.add(Dense(env.action_space.n, activation='linear'))
    # next_q_network.compile(loss='mse', optimizer=Adam(learning_rate=0.01))
    q_network = Sequential()
    q_network.add(Conv2D(32, 8, input_shape=env.observation_space.shape, strides=4, activation="relu"))
    q_network.add(Conv2D(64, 4, strides=4, activation="relu"))
    q_network.add(Conv2D(64, 3, strides=2, activation="relu"))
    q_network.add(Flatten())
    q_network.add(Dense(512, activation="relu"))
    q_network.add(Dense(env.action_space.n, activation="linear"))

    q_network.compile(loss='mse', optimizer=Adam(learning_rate=0.01))

    next_q_network = Sequential()
    next_q_network.add(Conv2D(32, 8, strides=4, activation="relu", input_shape=env.observation_space.shape))
    next_q_network.add(Conv2D(64, 4, strides=4, activation="relu"))
    next_q_network.add(Conv2D(64, 3, strides=2, activation="relu"))
    next_q_network.add(Flatten())
    next_q_network.add(Dense(512, activation="relu"))
    next_q_network.add(Dense(env.action_space.n, activation="linear"))

    next_q_network.compile(loss='mse', optimizer=Adam(learning_rate=0.01))

    next_q_network.set_weights(q_network.get_weights())

    rewards = []
    q_func = []

    for i_episode in range(n_episodes):
        observation = env.reset()
        observation = np.reshape(observation, [1, 210, 160, 3])
        counts = 0
        q_func.append(np.amax(q_network.predict(observation)))
        for i in range(1000):
            # env.render()

            # perform action, with eps we do random action
            if random.random() <= eps:
                action = env.action_space.sample()
            else:
                q_values = q_network.predict(observation)
                action = np.argmax(q_values[0])

            next_observation, reward, done, _ = env.step(action)
            counts += reward

            next_observation = np.reshape(next_observation, [1, 210, 160, 3])
            memory.append((observation, action, reward, next_observation, done))

            observation = next_observation

            if done:
                next_q_network.set_weights(q_network.get_weights())
                break

            if len(memory) <= batch_size:
                continue

            minibatch = random.sample(memory, batch_size)

            for b_observation, b_action, b_reward, b_next_observation, b_done in minibatch:

                target = q_network.predict(b_observation)

                if b_done:
                    target[0][b_action] = b_reward
                else:
                    t = next_q_network.predict(b_next_observation)
                    target[0][b_action] = b_reward + gamma * np.amax(t)

                q_network.fit(b_observation, target, epochs=1, verbose=0)

            # print("time: {}".format(counts))
            # print("reward: {}".format(reward))
            # print("state: {}".format(observation))
            # print("action: {}".format(action))
            # print("")
        print("Episode {} finished after {} timesteps".format(i_episode, counts))
        rewards.append(counts)
    return q_network, rewards, q_func


def print_graph(n=9):
    env = gym.make('Breakout-v0')
    _, rewards, q_funcs = deep_q_learning(env)
    for t in range(n - 1):
        _, reward, q_func = deep_q_learning(env)
        rewards = np.array(rewards) + np.array(reward)
        q_funcs = np.array(q_func) + np.array(q_funcs)
    rewards = rewards / n
    q_funcs = q_funcs / n

    xs = np.arange(1, len(rewards) + 1, 1)
    plt.plot(xs, rewards)
    plt.show()

    plt.plot(xs, q_funcs)
    plt.show()


def show_game(env, model):
    observation = env.reset()
    for t in range(10000):
        observation = np.reshape(observation, [1, 210, 160, 3])
        env.render()
        q_values = model.predict(observation)
        action = np.argmax(q_values[0])
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # env = gym.make('CartPole-v0')
    print_graph()
    # model, rewards = deep_q_learning(env)
    # show_game(env, model)
    # env.close()
