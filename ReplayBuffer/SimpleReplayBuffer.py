from ReplayBuffer.ReplayBuffer import ReplayBuffer


class SimpleReplayBuffer(ReplayBuffer):
    def __init__(self, env, replay_buffer_size=10**6):
        super().__init__(env, replay_buffer_size)

        self.data = []

    def sample(self, batch_size=None):
        return self.data

    def add(self, data, **kwargs):
        self.data.append(data)

    def clear(self):
        self.data = []
