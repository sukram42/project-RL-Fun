
import torch

from Agent import Agent
from PictureRecorder import PictureRecorder

import gym
import pybulletgym
import numpy as np
import random

from hyperparameter import HYPERPARAMETER, AGENTS
import helpers
#helpers.set_seed(int(HYPERPARAMETER.get('seed')))


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(int(HYPERPARAMETER.get('seed')))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


env = gym.make("InvertedPendulumPyBulletEnv-v0")
recorder = PictureRecorder(channel=3)

agent_name = HYPERPARAMETER.get('agent', 'random')
agent: Agent = AGENTS.get(agent_name)(env, HYPERPARAMETER.get(agent_name))

step = 0
info = {}
eval_period = HYPERPARAMETER.get('eval_period')


def print_log(_episode, _logs):
    print(f"EPISODE {_episode}  " + " | ".join(map(lambda x: f"{x[0]}: {x[1]}", _logs.items())))


for episode in range(HYPERPARAMETER.get("episodes")):
    s = env.reset()
    done = False

    for step in range(HYPERPARAMETER.get("max_steps")):
        a = agent.sample(s, ep=episode).detach()
        assert a is not None

        next_s, rew, done, _ = env.step([a.item()])
        agent.update_model(s=s, a=a, next_s=next_s, rew=rew, done=done)
        if done:
            break
        s = next_s

        if episode % eval_period == 0:
            # recorder.record(env)
            recorder.add(env.render(mode='rgb_array'))

    logs = agent.train_model()
    logs['steps'] = step

    print_log(episode, logs)
    if episode % eval_period == 0:
        # recorder.save(episode)
        recorder.save_movie(f"videos/video_ep_{episode}.gif")

if __name__ == '__main__':
    pass
