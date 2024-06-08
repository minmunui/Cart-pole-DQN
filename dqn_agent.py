import random
from collections import deque
from typing import List, Tuple

import gymnasium
import keras
import numpy as np
from keras.layers import Dense


class DQN:
    def __init__(
            self,
            env: gymnasium.Env,
            discount_factor: float = 0.99,
            learning_rate: float = 0.001,
            epsilon: float = 0.5,
            epsilon_decay: float = 0.999,
            epsilon_min: float = 0.01,
            batch_size: int = 64,
            replay_buffer_size: int = 100,
    ):
        self.env = env
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.init_epsilon = epsilon
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
        self.episode_max = 1000

    def add_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        inputs = []
        targets = []
        for state, action, reward, next_state, done in batch:
            if done:
                targets.append([reward])
            else:
                targets.append([reward + self.discount_factor * np.max(self.target_model.predict(np.array([next_state,]), verbose=0))])
            inputs.append(state)
        print(f"Inputs shape: {len(inputs)}")
        print(f"Targets shape: {len(targets)}")
        self.model.fit(np.array(inputs), np.array(targets), epochs=1, verbose=2)

    def train_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def build_model(self):
        model = keras.models.Sequential()
        model.add(Dense(24, input_shape=self.observation_space.shape, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(gymnasium.spaces.utils.flatdim(self.action_space), activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state, deterministic=False):
        if not deterministic and np.random.rand() <= self.init_epsilon:
            return self.env.action_space.sample()
        print(f"State: {state}")
        return np.argmax(self.model.predict(np.array([state,]), verbose=0))

    def learn(self, n_episode: int) -> list[tuple[int, float]]:
        self.epsilon = self.init_epsilon
        self.learn_history = []
        for _ in range(n_episode):
            print(f"Episode: {self.episode}")
            state = self.env.reset()[0]
            total_reward = 0
            step = 0
            done = False
            while not done:
                action = self.act(state)
                print(f"Step action: {action}")
                next_state, reward, done, _, _ = self.env.step(action)
                self.add_experience(state, action, reward, next_state, done)
                if len(self.replay_buffer) >= self.batch_size:
                    self.train()
                # if len(self.replay_buffer) >= self.replay_buffer.maxlen:
                #     self.train()
                total_reward += reward
                state = next_state
                step += 1
                if self.init_epsilon > self.epsilon_min:
                    self.init_epsilon *= self.epsilon_decay
                if step > self.episode_max:
                    done = True
            self.train_target_model()
            self.learn_history.append((total_reward, self.epsilon))
            self.episode += 1
        return self.learn_history
