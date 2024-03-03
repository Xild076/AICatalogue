import numpy as np
from random import random


class QLearning(object):
    def __init__(self, env, config):
        self.env = env

        self.num_states = env.observation_space
        self.num_actions = env.action_space
        self.learning_rate = config.get('learning_rate', 0.01)
        self.discount = config.get('discount', 0.99)
        self.exploration_rate = config.get('exploration_rate', 0.02)
        self.reward_list = []
        
        self.end = False
        self.pause = False

        self.model_init()

    def model_init(self):
        self.q_table = np.zeros([self.num_states, self.num_actions])
    
    def update_table(self, old_state, new_state, action, reward):
        init_value = self.q_table[old_state, action]
        init_best = np.max(self.q_table[new_state])
        new_value = (1 - self.learning_rate) * init_value + self.learning_rate * (reward + self.discount * init_best)
        self.q_table[old_state, action] = new_value
    
    def train(self, epochs, count_max):
        old_state = self.env.reset()
        done = False
        
        for epoch in epochs:
            counter = 0
            
            while not done and counter < count_max:
                counter += 1
                
                if random < self.exploration_rate:
                    action = random.randint(0, self.num_actions - 1)
                else:
                    action = np.argmax(self.q_table[old_state])
                
                state, reward, done, _ = self.env.step(action)
                self.update_table(old_state, state, action, reward)