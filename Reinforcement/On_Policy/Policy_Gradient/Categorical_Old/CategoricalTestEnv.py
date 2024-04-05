from random import random
import time
import numpy as np


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
            time.sleep(0.01)
            
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