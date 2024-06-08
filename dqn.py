import random
from collections import deque

import gymnasium
import keras
import numpy as np
from keras import Sequential
from keras.layers import Dense


class dqn_agent:

    def __init__(self,
                 env: gymnasium.Env,
                 discount_factor: float = 0.99,
                 learning_rate: float = 0.001,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.999,
                 epsilon_min: float = 0.01,
                 batch_size: int = 64,
                 replay_buffer_size: int = 1000
                 ):
        self.env = env
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen=replay_buffer_size)

        self.action_space = env.action_space
        self.observation_space = env.observation_space

        self.target_model = self.build_model()
        self.model = self.build_model()
        self.episode = 0

        self.learn_history = []

        self.train_start = 1000
        self.episode_max = 1000
        self.max_episode_reward = 1000

        self.min_train_end_reward_rate = 0.95
        self.current_episode_queue_max = 15


    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_shape=self.observation_space.shape, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(gymnasium.spaces.utils.flatdim(self.action_space), activation='linear'))
        print("Model Summary")
        model.summary()
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min
        mini_batch = random.sample(self.replay_buffer, self.batch_size)

        states = np.zeros((self.batch_size, self.observation_space.shape[0]))
        next_states = np.zeros((self.batch_size, self.observation_space.shape[0]))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        target = self.model.predict(states, verbose=0)
        target_val = self.target_model.predict(next_states, verbose=0)

        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * np.max(target_val[i])

        self.model.fit(states, target, epochs=1, verbose=0)

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return self.action_space.sample()
        else:
            # print(f"State: {state}")
            return np.argmax(self.model.predict(state, verbose=0))

    def add_sample(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def learn(self, episodes):

        current_episode_rewards = deque(maxlen=self.current_episode_queue_max)

        for episode in range(episodes):
            state = np.array([self.env.reset()[0]])
            # print(f"initial state: {state}")
            done = False
            total_reward = 0

            while not done:
#                 print(f"State: {state}")
                action = self.get_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.observation_space.shape[0]])
                self.add_sample(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state

                if len(self.replay_buffer) >= self.train_start:
                    self.train_model()

            self.update_target_model()
            self.learn_history.append((episode, total_reward, self.epsilon))
            current_episode_rewards.append(total_reward)
            if (len(current_episode_rewards) == self.current_episode_queue_max and
                np.mean(current_episode_rewards) > self.min_train_end_reward_rate * self.max_episode_reward):
                break

            print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {self.epsilon}, Buffer Size: {len(self.replay_buffer)}, {self.current_episode_queue_max}current_episodes_rewards_mean: {np.mean(current_episode_rewards)}")
        return self.learn_history
