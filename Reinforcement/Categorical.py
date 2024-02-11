import numpy as np
from random import random
from datetime import datetime
import time
import matplotlib.pyplot as plt
import threading
import winsound
import colorama
from enum import Enum
from functools import partial


class PolicyGradient(object):
    def __init__(self, env, learning_rate, hid_layers, activation):
        self.num_states = env.observation_space
        self.num_actions = env.action_space
        self.learning_rate = learning_rate
        self.hidden_layers = hid_layers
        self.model_init()
        self.activation = activation.value
    
    class Activation(Enum):  
        RELU = partial(lambda x: np.maximum(0, x))
        ELU = partial(lambda x, alpha: np.where(x >= 0, x, alpha * (np.exp(x) - 1)))
        SWISH = partial(lambda x: x / (1 + np.exp(-x)))
    
    def model_init(self):
        self.model = {
            1: np.random.randn(self.hidden_layers, self.num_states) / np.sqrt(self.num_actions) * self.learning_rate,
            2: np.random.randn(self.num_actions, self.hidden_layers) / np.sqrt(self.hidden_layers) * self.learning_rate,
        }
    
    def softmax(self, x):
        x = x.astype(np.float64)
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
        
    def policy_forward(self, state, alpha=0):
        hid = np.dot(self.model[1], state)
        hid = self.activation(hid, alpha) if alpha else self.activation(hid)
        logp = np.dot(self.model[2], hid)
        prob = self.softmax(logp)
        return prob, hid
    
    def discount_rewards(self, rewards):
        discounted_reward = np.zeros_like(rewards, dtype=np.float64)
        total_rewards = 0
        for t in reversed(range(0, len(rewards))):
            total_rewards = total_rewards * self.discount + rewards[t]
            discounted_reward[t] = total_rewards
        return discounted_reward
    
    def policy_backward(self, state_change, hid, gprob, alpha=0):
        dW2 = np.dot(hid.T, state_change)
        dh = np.dot(gprob, self.model[2])
        dh = self.activation(dh, 0) if alpha else self.activation(dh)
        dW1 = np.dot(dh.T, state_change)
        return {1: dW1, 2: dW2}