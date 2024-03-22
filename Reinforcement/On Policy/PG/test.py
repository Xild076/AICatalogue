import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import time


class Neuron(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.shape = (x, y)
        
        self.weight = np.random.randn(x, y)
    
    def update(self, add):
        self.weight += add
    
    @staticmethod
    def forward_pass(x, y, activation):
        return activation(np.dot(x, y.weight))


class Activation:
    @staticmethod
    def relu(x, derivative=False):
        if derivative:
            return np.where(x > 0, 1, 0)
        
        return np.maximum(0, x)
    
    @staticmethod
    def sigmoid(x, derivative=False):
        if derivative:
            sig_x = Activation.sigmoid(x)
            return sig_x * (1 - sig_x)
        pos_mask = (x >= 0)
        neg_mask = ~pos_mask
        z = np.zeros_like(x)
        z[pos_mask] = np.exp(-x[pos_mask])
        z[neg_mask] = np.exp(x[neg_mask])
        return 1 / (1 + z)
    
    @staticmethod
    def softmax(x, derivative=False):
        if derivative:
            return Activation.sigmoid(x, derivative)
        
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    @staticmethod
    def tanh(x, derivative=False):
        if derivative:
            return 1 - np.tanh(x) ** 2
        
        return np.tanh(x)


class Model(object):
    def __init__(self, *args):
        self.model = []
        list_arg = list(args[0])
        for arg in list_arg:
            if isinstance(arg, Neuron):
                self.model.append(arg)
        
        self.activation = []
        for i in range(len(self.model)):
            self.activation.append(list_arg[1+i*2])

    @staticmethod
    def fix_hid(hid):
        vhidden = []
        for layer in range(len(hid[0])):
            one_layer = []
            for i in range(len(hid)):
                one_layer.append(hid[i][layer])
            vhidden.append(np.array(one_layer))
        return vhidden
    
    def forward(self, x):
        passes = []
        for index, layer in enumerate(self.model):
            x = Neuron.forward_pass(x, layer, self.activation[index])
            passes.append(x)
        return passes
    
    def backward(self, vstate, vgrad, vhid):
        update_grads = []
        
        dh = vgrad.T
        vhid = Model.fix_hid(vhid)
        for i in range(len(vhid)):
            dw = np.dot(dh, vhid[-(i+1)]).T
            dh = self.activation[-(i+2)](np.dot(self.model[-(i+1)].weight, dh), derivative=True)
            update_grads.insert(0, dw)
        update_grads.insert(0, np.dot(dh, vstate).T)
        
        return update_grads
        

class Policy(object):
    def __init__(self, config, *args):
        self.lr = config.get('lr', 0.01)
        self.discount = config.get('discount', 0.99)
        
        self.model = Model(args)
        
    def discount_rewards(self, rewards):
        discounted_reward = np.zeros_like(rewards, dtype=np.float64)
        total_rewards = 0
        for t in reversed(range(0, len(rewards))):
            total_rewards = total_rewards * self.discount + rewards[t]
            discounted_reward[t] = total_rewards
        return discounted_reward
    
    def sample_action(self, action_prob, randomness):
        
        return NotImplementedError
    
    def action_grad(self, action, action_prob):
        
        return NotImplementedError

    def train(self, config):
        epochs = config.get('epochs', 1000)
        batch_size = config.get('batch', 10)
        env = config.get('env')
        
        reward_list = []
        
        for epoch in range(epochs):

            grad_buffer = [np.zeros_like(neuron.weight) for neuron in self.model.model]
            rmsprop = [np.zeros_like(neuron.weight) for neuron in self.model.model]


            sstate, shidden, sgrads, srewards = [], [], [], []

            state = env.reset()
            old_state = 0
            done = False
            counter = 0
            reward_sum = 0

            while not done:
                counter += 1
                calc_state = state - old_state

                passes = self.model.forward(calc_state)
                prob = passes.pop(-1)

                action = self.sample_action(prob)

                sstate.append(calc_state)
                shidden.append(passes)
                sgrads.append(self.action_grad(action, prob))

                state, reward, done, _ = env.step(action)
                srewards.append(reward)

                reward_sum += reward

            vstate = np.vstack(sstate)
            vhidden = shidden
            vgrads = np.vstack(sgrads)
            vrewards = np.vstack(srewards)

            discounted_vrew = self.discount_rewards(vrewards)
            discounted_vrew -= (np.mean(discounted_vrew)).astype(np.float64)
            discounted_vrew /= ((np.std(discounted_vrew)).astype(np.float64) + 1e-8)

            vgrads *= discounted_vrew
            grad = self.model.backward(vstate, vgrads, vhidden)
            grad_buffer = [gb + g for gb, g in zip(grad_buffer, grad)]

            if epoch % batch_size == 0:
                for i in range(len(self.model.model)):
                    g = grad_buffer[i]
                    rmsprop[i] = 0.99 * rmsprop[i] + (1-0.99) * g**2
                    grad_add = g * self.lr / (np.sqrt(rmsprop[i] + 1e-5))
                    grad_buffer[i] = np.zeros_like(self.model.model[i].weight)
                    self.model.model[i].update(grad_add)

            reward_list.append(reward_sum)
            print(epoch, reward_sum)

        plt.figure(figsize=(10, 6))
        sns.lineplot(x=range(len(reward_list)), y=reward_list, color='blue', label='Rewards', linewidth=1, alpha=0.6)

        x = np.array(range(len(reward_list)))
        y = np.array(reward_list)
        degree = 20
        coefficients = np.polyfit(x, y, degree, rcond=None)
        coefficients = np.polyfit(x, y, degree)
        poly = np.poly1d(coefficients)
        y_fit = poly(x)

        plt.plot(x, y_fit, color='blue', linestyle='-', label='Average Growth')

        plt.xlabel('Time')
        plt.ylabel('Reward')
        plt.title('Reward Growth Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return reward_list

class Categorical(Policy):
    def __init__(self, config, *args):
        super().__init__(config, *args)
    
    def sample_action(self, x):
        if random.random() < 0.1:
            return np.random.randint(0, len(x))
        return np.argmax(x)
    
    def action_grad(self, action, action_prob):
        aoh = np.zeros(len(action_prob))
        aoh[action] = 1
        return aoh - action_prob


class TestEnv(object):
    def __init__(self):
        self.observation_space = 3
        self.action_space = 3
        self.numbers = [random.random() * 10, random.random() * 10, random.random() * 10]
        self.counter = 0
    
    def reset(self):
        self.numbers = [random.random() * 10, random.random() * 10, random.random() * 10]
        self.counter = 0
        return self.get_state()
            
    def get_state(self):
        self.numbers = [random.random() * 10, random.random() * 10, random.random() * 10]
        return np.array(self.numbers)

    def step(self, action, test=False):
        total = np.sum(self.numbers)
        if action == np.argmax(self.numbers):
            reward = 1
        else:
            reward = -np.abs(action - np.argmax(self.numbers))
        if test:
            print('Action', action)
            print('Total', total)
            print('Reward', reward)
            
        self.counter += 1
        return self.get_state(), reward, self.counter == 25, None
  
    def render(self):
        print(self.numbers, np.sum(self.numbers))


alg = Categorical({'lr': 0.0001}, Neuron(3, 100), Activation.relu, Neuron(100, 3), Activation.softmax)
alg.train({'epochs': 10000, 'batch': 5, 'env': TestEnv()})