
class ReplayBuffer:

    def __init__(self,  env, replay_buffer_size: int = 10 ** 6):
        self.env = env
        self.replay_buffer_size = replay_buffer_size
        self.full = False
        self.idx = 0

    def __len__(self):
        return self.replay_buffer_size if self.full else self.idx

    def sample(self, batch_size):
        raise NotImplementedError()

    def add(self, **kwargs):
        raise NotImplementedError()
