from collections import deque

import gymnasium
import keras


class DQN:
    def __init__(
            self,
            env: gymnasium.Env,
            discount_factor: float = 0.99,
            learning_rate: float = 0.001,
            epsilon: float = 1.0,
            epsilon_decay: float = 0.995,
            epsilon_min: float = 0.01,
            batch_size: int = 64,
            n_step: int = 1,
            replay_buffer_size: int = 1000
    ):
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.n_step = n_step

        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.model = self.build_model()

        self.state_shape = env.observation_space.shape
        self.action_shape = env.action_space.shape

    def build_model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(24, input_shape=(self.state_shape,), activation='relu'))
        model.add(keras.layers.Dense(24, activation='relu'))
        model.add(keras.layers.Dense(self.action_, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model
