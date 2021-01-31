import time


class Agent:

    def __init__(self, env, param):
        self.env = env
        self.param = param

    def sample(self, state, **kwargs):
        raise NotImplementedError("The sample function of this agent is not implemented yet")

    def train_model(self, **kwargs):
        tic = time.time()

        res = self._train_model(**kwargs)
        assert type(res) == dict

        toc = time.time()
        res['dur'] = f"{toc-tic:.2f} s"
        return res

    def _train_model(self, **kwargs):
        raise NotImplementedError("The training of this agent is not implemented yet")

    def update_model(self, s, a, rew, next_s, done):
        pass
