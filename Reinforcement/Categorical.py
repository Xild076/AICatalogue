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
    def __init__(self, env, learning_rate, hid_layers, activation, optimization, alpha=0):
        self.env = env
        self.num_states = env.observation_space
        self.num_actions = env.action_space
        self.learning_rate = learning_rate
        self.hidden_layers = hid_layers
        self.model_init()
        self.activation = activation.value
        self.optimization = optimization.value
        self.alpha = alpha
        self.discount = 0.99
        self.beta1 = 0.99
        self.beta2 = 0.99
    
    class Activation(Enum):  
        RELU = partial(lambda x: np.maximum(0, x))
        ELU = partial(lambda x, alpha: np.where(x >= 0, x, alpha * (np.exp(x) - 1)))
        SWISH = partial(lambda x: x / (1 + np.exp(-x)))
    
    class Optimization(Enum):
        RMSPROP = partial(lambda g, c1, c2, b1, b2, lr, epoch: PolicyGradient.Optimization.rmsprop(g, c1, c2, b1, b2, lr, epoch))
        SGD = partial(lambda g, c1, c2, b1, b2, lr, epoch: PolicyGradient.Optimization.sgd(g, c1, c2, b1, b2, lr, epoch))
        ADAM = partial(lambda g, c1, c2, b1, b2, lr, epoch: PolicyGradient.Optimization.adam(g, c1, c2, b1, b2, lr, epoch))
        NADAM = partial(lambda g, c1, c2, b1, b2, lr, epoch: PolicyGradient.Optimization.nadam(g, c1, c2, b1, b2, lr, epoch))
        
        def rmsprop(grad, cache1, cache2, beta1, beta2, lr, epoch):
            rc = beta1 * cache1 + (1 - beta1) * grad**2
            return (lr * grad / (np.sqrt(rc) + 1e-8)), rc, 0
        
        def sgd(grad, cache1, cache2, beta1, beta2, lr, epoch):
            return - lr * grad, 0, 0
        
        def adam(grad, cache1, cache2, beta1, beta2, lr, epoch):
            c1 = beta1 * cache1 + (1 - beta1) * grad
            c2 = beta2 * cache2 + (1 - beta1) * grad**2
            c1_corrected = c1 / (1 - beta1 ** epoch)
            c2_corrected = c2 / (1 - beta2 ** epoch)
            grad_b_add = - lr * c1_corrected / (np.sqrt(c2_corrected) + 1e-8)
            return grad_b_add, c1, c2

        def nadam(grad, cache1, cache2, beta1, beta2, lr, epoch):
            c1 = beta1 * cache1 + (1 - beta1) * grad
            c2 = beta2 * cache2 + (1 - beta2) * grad**2
            c1_corrected = c1 / (1 - beta1 ** epoch)
            c2_corrected = c2 / (1 - beta2 ** epoch)
            grad_b_add = - lr * (beta1 * c1_corrected + (1 - beta1) * grad) / (np.sqrt(c2_corrected) + 1e-8)
            return grad_b_add, c1, c2
            
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

    def train(self, epochs, batch_size, count_max):
        for epoch in range(epochs):
            grad_buffer = {k: np.zeros_like(v) for k, v in self.model.items()}
            self.cache_1 = {k: np.zeros_like(v) for k, v in self.model.items()}
            self.cache_2 = {k: np.zeros_like(v) for k, v in self.model.items()}
            sstate, shidden, sgrads, srewards = [], [], [], []
            
            state = self.env.reset()
            old_state = 0
            done = False
            counter = 0
            reward_sum = 0
            
            while not done and counter <= count_max:
                counter += 1
                calc_state = state - old_state
                
                prob, hid = self.policy_forward(calc_state, self.alpha)
                action = np.random.choice(np.arange(self.num_actions))
                
                aoh = np.zeros(self.num_actions)
                aoh[action] = 1
                
                sstate.append(calc_state)
                shidden.append(hid)
                sgrads.append(aoh - prob)
                
                state, reward, done, _ = self.env.step(action)
                srewards.append(reward)
                
                reward_sum += reward
                
            vstate = np.vstack(sstate)
            vhidden = np.vstack(shidden)
            vgrads = np.vstack(sgrads)
            vrewards = np.vstack(srewards)
            
            discounted_vrew = self.discount_rewards(vrewards)
            discounted_vrew -= (np.mean(discounted_vrew)).astype(np.float64)
            discounted_vrew /= ((np.std(discounted_vrew)).astype(np.float64) + 1e-8)

            vgrads *= discounted_vrew
            grad = self.policy_backward(vstate, vhidden, vgrads)
            for k in self.model: 
                grad_buffer[k] += grad[k]
        
            if epoch % batch_size == 0:
                for k, v in self.model.items():
                    g = grad_buffer[k]
                    grad_add, self.cache_1[k], self.cache_2[k] = self.optimization(g, self.cache_1[k], self.cache_2[k], 
                                                                                   self.beta1, self.beta2, self.learning_rate, epoch)
                    self.model[k] += grad_add
                    grad_buffer[k] = np.zeros_like(v)