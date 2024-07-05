from nn import MLP
from engine import value
from optim import SGD
from dist import Categorical
import random
import numpy as np


class PPO:
    def __init__(self, env, actor_net, value_net, optim_class, lr):
        self.env = env
        self.actor_net = actor_net
        self.value_net = value_net
        self.optim = optim_class(actor_net.parameters() + value_net.parameters(), lr)
        self.gamma = 0.99
        self.kl_coeff = 0.2
        self.vf_coeff = 0.5
    
    def pick_sample_and_logp(self, s):
        logits = self.actor_net(s)
        probs = logits.softmax()
        c = Categorical(probs=probs)
        