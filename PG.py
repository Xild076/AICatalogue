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


class PG(object):
    def __init__(self, layers, activ, optm, lr, discount, categorical=True):
        self.mlp = MLP(layers, activ)
        print(self.mlp.parameters())
        self.optm = SGD(self.mlp.parameters(), lr)
        self.c = categorical
        self.discount = discount
    
    def forward(self, s):
        return Categorical(self.mlp(s))
    
    def discount_r(self, r):
        dr = Array.zeros_like(r)
        ra = 0
        for t in reversed(range(len(r))):
            if r[t] !=0: ra = 0
            ra = ra * self.discount + r[t]
            dr[t] = ra
        return dr
    
    def train(self, env, epochs):
        for epoch in range(epochs):
            rewards = []
            actions = []
            states  = []
            logits = []
            
            o_s = 0
            s = env.reset()
            done = False
            while not done:
                e_s = s - o_s
                p = self.forward(e_s)
                if self.c:
                    o_s = s
                states.append(e_s)
                
                a = p.sample()
                
                s, r, done, _ = env.step(a)
                
                actions.append(a)
                rewards.append(r)
                logits.append(p)
            
            dr = self.discount_r(rewards)
            log_p = Array(logit.log_prob(action) for logit, action in zip(logits, actions))
            loss = -sum(dr * log_p)
            
            self.optm.zero_grad()
            loss.backward()
            self.optm.step()
            
            print(f"Episode {epoch + 1} | Reward: {sum(rewards)} | Loss: {loss}")
            

import random

class SimpleEnv:
    def __init__(self):
        self.state = 0  # Initial state
        self.steps = 0  # Counter for steps
        self.max_steps = 10  # Maximum steps in an episode

    def reset(self):
        """Reset the environment to an initial state."""
        self.state = 0
        self.steps = 0
        return self.state

    def step(self, action):
        """Take an action and return the new state, reward, and done flag."""
        self.steps += 1
        reward = self.reward_function(action)
        self.state = self.next_state(action)
        done = self.steps >= self.max_steps  # Episode ends after max_steps
        return self.state, reward, done, {}

    def next_state(self, action):
        """Simple state transition based on the action."""
        # State transitions can be arbitrary; here it's simple increment or decrement
        if action == 0:
            return self.state - 1
        elif action == 1:
            return self.state + 1
        else:
            return self.state

    def reward_function(self, action):
        """Define rewards for actions."""
        # Simple reward function: reward +1 for action 1, -1 for action 0
        if action == 0:
            return -1
        elif action == 1:
            return 1
        else:
            return 0  # Default reward for other actions (if any)



# Define a simple MLP architecture for the policy
layers = [1, 2, 2]  # Input size = 1, hidden layer with 4 neurons, output size = 2
activ = ['relu', '']      # Activation function, assuming 'relu' is defined in your NN library
optm = SGD          # Using SGD optimizer
lr = 0.1           # Learning rate
discount = 0.99     # Discount factor

# Instantiate the Policy Gradient agent
pg_agent = PG(layers, activ, optm, lr, discount)

# Instantiate the environment
env = SimpleEnv()

# Train the agent for a number of episodes
pg_agent.train(env, epochs=100)
