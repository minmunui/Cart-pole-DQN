import gymnasium
import matplotlib.pyplot as plt
import numpy as np

from dqn import dqn_agent

REPLAY_BUFFER_SIZE = 5000
BATCH_SIZE = 64

env = gymnasium.make('CartPole-v1')
env.metadata = {'render_modes': ['human']}
dqn_agent = dqn_agent(env, batch_size=128, replay_buffer_size=5000)
learn_result = dqn_agent.learn(500)

data = np.array(learn_result)

X = data[:, 0]
rewards = data[:, 1]
epsilon = data[:, 2]

fig, ax1 = plt.subplots()

ax1.plot(X, rewards, 'g-')
ax1.set_ylim(0, 1100)
ax1.set_xlabel('episode')
ax1.set_ylabel('reward of episode', color='g')

ax2 = ax1.twinx()
ax2.plot(X, epsilon, 'b-')
ax2.set_ylim(0, 1.1)
ax2.set_ylabel('epsilon', color='b')

plt.title(f"DQN Agent Learning batch_size={BATCH_SIZE}, replay_buffer_size={REPLAY_BUFFER_SIZE}")
plt.savefig("DQN Agent Learning batch_size={BATCH_SIZE}, replay_buffer_size={REPLAY_BUFFER_SIZE}")
plt.show()
