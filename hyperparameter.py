from Agent import Agent
from REINFORCE.REINFORCE import REINFORCE
from RandomAgent.RandomAgent import RandomAgent

HYPERPARAMETER = {
    "seed": 10,
    "agent": "REINFORCE",
    "episodes": 10000,
    "max_steps": 500,
    "eval_period": 500,

    "REINFORCE": {
        "lr": 0.005,
        "hidden_dim": 128,
        "gamma": 0.98
    }
}

AGENTS = {
    "random": RandomAgent,
    "REINFORCE": REINFORCE
}
