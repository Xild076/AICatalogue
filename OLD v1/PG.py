import cProfile
import pstats
from engine import Value, Array
from nn import MLP, Layer, Neuron
from loss import CrossEntropy
from optim import SGD, RMSProp, ADAM
import math, random
from cd import Categorical
import copy
import time
import numpy as np
import gym


class PG:
    def __init__(self, layers, activ, optm, lr, discount, categorical=True):
        self.mlp = MLP(layers, activ)
        self.optm = optm(self.mlp.parameters(), lr)
        self.categorical = categorical
        self.discount = discount
    
    def forward(self, s):
        return Categorical(self.mlp(s))
    
    def discount_r(self, r):
        dr = Array.zeros_like(r)
        for i in range(len(r)):
            ds = 0
            for j in range(i, len(r)):
                ds += r[j] * (self.discount ** (j - i))
            dr[i] = ds
        return dr
    
    def train(self, env, epochs):
        for epoch in range(epochs):
            rewards = []
            actions = []
            states = []
            
            o_s = 0
            s = env.reset()[0]
            
            done = False
            while not done:
                e_s = s - o_s
                p = self.forward(e_s)
                if self.categorical:
                    o_s = s
                states.append(e_s)
                
                a = p.sample(0.1)
                
                s, r, done, _, _ = env.step(a)
                
                states.append(s)
                actions.append(a)
                rewards.append(r)
            
            logits = [self.forward(state) for state in states]
            sampler = [Categorical(logit) for logit in logits]
            log_prob = Array(-sample.log_prob(action) for sample, action in zip(sampler, actions))
            loss = sum(log_prob * dr)
            
            self.optm.zero_grad()
            loss.backward()
            self.optm.step()
            
            print(f"Episode {epoch + 1} | Reward: {sum(rewards)} | Loss: {loss}")
        env.close()

env_name = 'CartPole-v1'
env = gym.make(env_name)


p = PG([4, 32, 2], ['relu', ''], SGD, 0.01, 0.99, False)
p.train(env, 100)