import PG
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from random import random


class TestEnv(object):
    def __init__(self):
        self.observation_space = 3
        self.action_space = 3
        self.numbers = [random(), random(), random()]
        self.counter = 0
    
    def reset(self):
        self.numbers = [random(), random(), random()]
        self.counter = 0
        return self.get_state()
            
    def get_state(self):
        self.numbers = [random(), random(), random()]
        return np.array(self.numbers)

    def step(self, action, test=False):
        total = np.sum(self.numbers)
        if action == np.argmax(self.numbers):
            reward = 1
        else:
            reward = -1
        if test:
            print('Action', action)
            print('Total', total)
            print('Reward', reward)
            
        self.counter += 1
        return self.get_state(), reward, self.counter == 100, None
  
    def render(self):
        print(self.numbers, np.sum(self.numbers))


class TestEnvironment:
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.generate_data()

    def generate_data(self):
        a = 2
        b = -3
        c = 1

        self.x_train = np.random.uniform(-10, 10, self.num_samples)

        self.y_train = a * self.x_train**2 + b * self.x_train + c + np.random.normal(0, 5, self.num_samples)

    def get_train(self):
        return self.x_train, self.y_train


"""model = PG.Model({'lr': 0.00001, 'optm': 'sgd', 'type': 'loss', 'lt': 'mse'}, [100, 200, 100], ['', '', ''])
o, _ = model.train({'epochs': 5000, 'env': TestEnvironment()})

model.visualize(o, _)"""


model = PG.Model({'lr': 0.001, 'optm': 'rmsprop', 'type': 'reward', 'lt': 'mse', 'cont': False}, [3, 200, 3], ['', '', '', ''])
o, _ = model.train({'epochs': 20000, 'env': TestEnv()})

model.visualize(o, _)