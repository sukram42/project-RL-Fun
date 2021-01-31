import torch
from torch import nn, optim, Tensor

from Agent import Agent
from ReplayBuffer.ExtendedReplayBuffer import ReplayBuffer
from ReplayBuffer.SimpleReplayBuffer import SimpleReplayBuffer


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, lr=0.002, gamma=0.98, hidden_dim=128):
        super(PolicyNetwork, self).__init__()

        self.gamma = gamma
        self.forward_1 = nn.Linear(obs_dim, hidden_dim)
        self.forward_2 = nn.Linear(hidden_dim, hidden_dim)

        # Mean network
        self.mean = nn.Linear(hidden_dim, 1)
        # std network
        self.std = nn.Linear(hidden_dim, 1)
        self.opt = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = torch.Tensor(x)
        x = self.forward_1(x)
        x = nn.ReLU()(x)
        # x = self.forward_2(x)
        # x = nn.ReLU()(x)

        mean = torch.tanh(self.mean(x))
        std = nn.Softmax()(self.std(x))
        return mean, std

    def train_model(self, buffer_data):
        c_rew = 0

        self.opt.zero_grad()

        loss = 0
        log_loss = []
        # we do not want to iterate over the last, because it made us dead
        for r, prob in buffer_data[::-1]:
            c_rew = r + self.gamma * c_rew
            loss = -prob * c_rew
            loss.backward()
            log_loss.append(loss)
        self.opt.step()

        avg_loss = (sum(log_loss)/len(log_loss))/len(buffer_data[::-1])

        return avg_loss


class REINFORCE(Agent):
    log_p: Tensor = None

    def __init__(self, env, param):
        if param is None:
            print("WARNING: Parameter for REINFORCE NOT GIVEN")
        super().__init__(env, param)
        self.buffer = SimpleReplayBuffer(env)
        self.pi = PolicyNetwork(obs_dim=env.observation_space.shape[0],
                                lr=param.get('lr'),
                                gamma=param.get('gamma'),
                                hidden_dim=param.get('hidden_dim')
                                )

    def _train_model(self, **kwargs):
        data = self.buffer.sample()
        avg_loss = self.pi.train_model(data)
        self.buffer.clear()
        return {"loss": f"{avg_loss.item(): 0.4f}"}

    def update_model(self, rew, **kwargs):
        assert self.log_p is not None
        self.buffer.add((rew, self.log_p))
        self.log_p=None

    def sample(self, state, **kwargs):
        _avg, _std = self.pi(state)
        dist = torch.distributions.normal.Normal(_avg, _std)
        a = dist.sample()

        self.log_p = dist.log_prob(a)
        return a
