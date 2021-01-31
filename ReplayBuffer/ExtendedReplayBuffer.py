import numpy as np

from ReplayBuffer.ReplayBuffer import ReplayBuffer


class ExtendedReplayBuffer(ReplayBuffer):

    def __init__(self, env, replay_buffer_size: int = 10 ** 6, reward_dim=1, done_dim=1):
        super().__init__(env, replay_buffer_size)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.reward = np.empty((replay_buffer_size, reward_dim))
        self.state = np.empty((replay_buffer_size, state_dim))
        self.new_state = np.empty((replay_buffer_size, state_dim))
        self.done = np.empty((replay_buffer_size, done_dim))
        self.action = np.empty((replay_buffer_size, action_dim))

    def add(self, state, action, new_state, reward, done):
        assert state.shape[0] == self.state.shape[1]
        self.state[self.idx] = state

        assert action.shape[0] == self.action.shape[1]
        self.action[self.idx] = action

        assert new_state.shape[0] == self.new_state.shape[1]
        self.new_state[self.idx] = new_state

        self.reward[self.idx] = reward
        self.done[self.idx] = done

        # Update the index
        self.idx = (self.idx + 1) % self.replay_buffer_size
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        _indices = np.random.randint(low=0,
                                     high=self.replay_buffer_size if self.full else self.idx,
                                     size=batch_size)

        return self.state[_indices], self.action[_indices], self.new_state[_indices], self.reward[_indices], \
               self.done[_indices]
