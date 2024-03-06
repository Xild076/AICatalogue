import numpy as np
from random import random
import time
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import threading



class BasicGradient(object):
    def __init__(self, env, config):
        self.env = env
        self.num_states = env.observation_space
        self.num_actions = env.action_space
        self.learning_rate = config.get('learning_rate', 0.01)
        self.discount = config.get('discount', 0.99)
        
        self.end = False
        self.pause = False
        
        self.model_init()
    
    def model_init(self):
        self.model = np.random.randn(self.num_actions, self.num_states)
    
    def policy_forward(self, state):
        mean = np.dot(state, self.model)
        action_grad = np.random.normal(loc=mean)
        return action_grad, mean
    
    def policy_backward(self, state, action_grad):
        action_grad = np.expand_dims(action_grad, axis=1)
        grad = action_grad * state
        return grad
    
    def train(self, epochs, batch_size, count_max):
        self.reward_list = []
        for epoch in range(epochs):
            
            if self.end:
                break
            
            while self.pause:
                time.sleep(0.1)
            
            state = self.env.reset()
            old_state = 0
            done = False
            counter = 0
            reward_sum = 0

            while not done and counter < count_max:
                counter += 1
                calc_state = state - old_state

                prob, _ = self.policy_forward(calc_state)

                state, reward, done, _ = self.env.step(prob)
                grad_add = self.policy_backward(state, prob) * reward * self.learning_rate
                
                self.model += grad_add
                
                reward_sum += reward
                

            print(reward_sum)
            self.reward_list.append(reward_sum)

    def render(self):
            def better_render(mogus):
                try:
                    plt.cla()
                    plt.plot(np.arange(len(self.reward_list)), self.reward_list, label='Reward')
                except:
                    pass

            ani = FuncAnimation(plt.figure(), better_render, interval=2000, cache_frame_data=False)
            plt.tight_layout()
            plt.show()

    def train_render(self, episodes, batch_size, cmax):
        func1 = threading.Thread(target=self.train, args=(episodes, batch_size, cmax))
        func1.start()
        self.render()
        func1.join()



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
        reward = 1 - abs(total - np.sum(action))
        if test:
            print('Action', action)
            print('Total', total)
            print('Reward', reward)
            time.sleep(0.01)

        self.counter += 1
        return self.get_state(), reward, self.counter == 100, None

    def render(self):
        print(self.numbers, np.sum(self.numbers))


pg = BasicGradient(TestEnv(), {'learning_rate': 0.00001})
pg.train_render(10000, 10, 100)