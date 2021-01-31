import torch

from Agent import Agent


class RandomAgent(Agent):

    def __init__(self, env):
        super().__init__(env)

    def sample(self, state, **kwargs):
        return self.env.action_space.sample()

    def _train_model(self, **kwargs):
        return {"test": 10, "hallo": 40}
