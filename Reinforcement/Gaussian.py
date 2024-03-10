"""action = np.array([1, 2, 3])[np.newaxis]
state = np.array([3, 2])[np.newaxis]
theta = np.random.randn(3, 2)

def softmax(x):
    x = x.astype(np.float64)
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

print('theta', theta)
reward = -1
lr = 0.001

print('theta shape', theta.shape)
print('state shape', state.shape)
print('action shape', action.shape)
print('state.t shape', state.T.shape)

forward = np.dot(theta, state.T)
print('forward', forward)
print('forward shape', forward.shape)

grad = np.dot(forward, state)

print('grad', grad)

theta += grad * lr * reward

forward = np.dot(theta, state.T)
print('forward', forward)
print('forward shape', forward.shape)
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
from random import random
import time

class GaussianGradient:
    def __init__(self, env, config):
        self.env = env
        self.num_states = env.observation_space
        self.num_actions = env.action_space
        self.learning_rate = config.get('learning_rate', 0.1)
        self.continuous = config.get('continuous', False)
        self.model_init()

    def model_init(self):
        self.model = np.random.randn(self.num_actions, self.num_states) / np.sqrt(self.num_states) * self.learning_rate

    def policy_forward(self, state):
        prob =  np.dot(self.model, state)
        action = np.random.normal(loc=prob, scale=self.learning_rate**2)
        action = np.clip(action, -1e5, 1e5)
        return action

    def policy_backward(self, action, state):
        return np.outer(action, state)

    def softmax(self, x):
        x = x.astype(np.float64)
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def train(self, epochs, count_max):
        self.reward_sum_list = []

        for epoch in range(epochs):
            state = self.env.reset()
            old_state = np.zeros_like(state) if self.continuous else 0
            done = False
            counter = 0
            reward_sum = 0
            reward_list = []
            state_list = []

            while not done and counter < count_max:
                counter += 1
                calc_state = state - old_state
                state_list.append(state.copy())
                if self.continuous:
                    old_state = state.copy()

                action = self.policy_forward(calc_state)
                print('act', action)
                print('state', state)
                state, reward, done, _ = self.env.step(action)
                reward_list.append(reward)
                reward_sum += reward

                averaged_reward = np.array(reward_list) - np.mean(reward_list)

                grad = self.policy_backward(action, state_list[-1])
                grad = np.nan_to_num(grad)
                grad = np.clip(grad, -1e10, 1e10)
                self.model -= grad * self.learning_rate * averaged_reward[-1]

            self.reward_sum_list.append(reward_sum)
            print(reward_sum)

    def render(self):
        def better_render(mogus):
            done = False
            while not done:
                try:
                    plt.cla()
                    plt.plot(np.arange(len(self.reward_sum_list)), self.reward_sum_list, label='Reward')
                    done = True
                except:
                    done = False
        ani = FuncAnimation(plt.figure(), better_render, interval=2000, cache_frame_data=False)
        plt.tight_layout()
        plt.show()

    def train_render(self, episodes, cmax):
        func1 = threading.Thread(target=self.train, args=(episodes, cmax))
        func1.start()
        self.render()
        func1.join()

class TestEnv(object):
    def __init__(self):
        self.observation_space = 3
        self.action_space = 3
        self.numbers = [50*random(), 50*random(), 50*random()]
        self.counter = 0

    def reset(self):
        self.numbers = [50*random(), 50*random(), 50*random()]
        self.counter = 0
        return self.get_state()

    def get_state(self):
        self.numbers = [50*random(), 50*random(), 50*random()]
        return np.array(self.numbers)

    def step(self, action, test=False):
        total = np.sum(self.numbers)
        reward = 10 - abs(self.numbers[0] - action[0]) - abs(self.numbers[1] - action[1]) - abs(self.numbers[2] - action[2])
        if test:
            print('Action', action)
            print('Total', total)
            print('Reward', reward)
            time.sleep(0.01)

        self.counter += 1
        return self.get_state(), reward, self.counter == 100, None

    def render(self):
        print(self.numbers, np.sum(self.numbers))
 

GG = GaussianGradient(TestEnv(), {})
GG.train_render(20, 10)