import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import time


class Neuron(object):
    def __init__(self, x, y, lr=0.01):
        self.x = x
        self.y = y
        self.shape = (x, y)
        
        self.weight = np.random.randn(x, y) / np.sqrt(y) * lr
    
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
        return 1/(1 + np.exp(-x)) 
    
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
    def __init__(self, args):
        self.model = []
        self.activation = []
        
        # Extract neurons and activation functions from arguments
        for i in range(len(args)):
            if isinstance(args[i], Neuron):
                self.model.append(args[i])
                # Check if activation function is provided for this layer
                if i + 1 < len(args) and callable(args[i + 1]):
                    self.activation.append(args[i + 1])

    def forward(self, x, test=False):
        passes = []
        for i in range(len(self.model)):
            x = Neuron.forward_pass(x, self.model[i], self.activation[i]) if i < len(self.activation) else Neuron.forward_pass(x, self.model[i], Activation.sigmoid)
            passes.append(x)
            if test: print(x)
        return passes

    def backward(self, vstate, vgrad, vhid):
        update_grads = []
        dh = vgrad.T
        vhid = Model.fix_hid(vhid)
        
        for i in range(len(vhid)):
            dw = np.dot(dh, vhid[-(i + 1)]).T
            dh = np.dot(self.model[-(i + 1)].weight, dh) * self.activation[-(i + 2)](vhid[-(i + 1)], derivative=True).T
            update_grads.insert(0, dw)
        
        update_grads.insert(0, np.dot(dh, vstate).T)
        
        return update_grads

    @staticmethod
    def fix_hid(hid):
        vhidden = []
        for layer in range(len(hid[0])):
            one_layer = []
            for i in range(len(hid)):
                one_layer.append(hid[i][layer])
            vhidden.append(np.array(one_layer))
        return vhidden
        

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
            discounted_vrew -= np.mean(discounted_vrew)
            discounted_vrew /= np.std(discounted_vrew) + 1e-8
            vgrads *= discounted_vrew
            grad = self.model.backward(vstate, vgrads, vhidden)
            
            for i in range(len(self.model.model)):
                g = grad[i]
                rmsprop[i] = 0.99 * rmsprop[i] + (1 - 0.99) * g ** 2
                grad_add = g * self.lr / (np.sqrt(rmsprop[i]) + 1e-5)
                grad_buffer[i] += grad_add
            
            if epoch % batch_size == 0:
                for i in range(len(self.model.model)):
                    self.model.model[i].update(grad_buffer[i] / batch_size)
                    grad_buffer[i] = np.zeros_like(self.model.model[i].weight)
            
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
        if random.random() < 0.05:
            return np.random.randint(0, len(x))
        return np.argmax(x)
    
    def action_grad(self, action, action_prob):
        aoh = np.zeros(len(action_prob))
        aoh[action] = 1
        return aoh - action_prob


class TestEnv(object):
    def __init__(self):
        self.observation_space = 4
        self.action_space = 4
        self.numbers = [random.random() * 10, random.random() * 10, random.random() * 10, random.random() * 10]
        self.counter = 0
    
    def reset(self):
        self.numbers = [random.random() * 10, random.random() * 10, random.random() * 10, random.random() * 10]
        self.counter = 0
        return self.get_state()
            
    def get_state(self):
        return np.array(self.numbers)

    def step(self, action, test=False):
        reward = ((0.3 - abs(action - np.argmax(self.numbers))) * 2) ** 3
        if np.argmax(self.numbers) == action:
            reward += 5
        if test:
            print('Action', action)
            print('numbers', self.numbers)
            print('Reward', reward)
            
        self.counter += 1
        self.numbers = [random.random() * 10, random.random() * 10, random.random() * 10, random.random() * 10]
        return self.get_state(), reward, self.counter == 50, None
  
    def render(self):
        print(self.numbers, np.sum(self.numbers))


alg = Categorical({'lr': 0.01}, Neuron(4, 20, 0.01), Activation.relu, Neuron(20, 4, 0.01), Activation.softmax)
alg.train({'epochs': 10000, 'batch': 10, 'env': TestEnv()})

t = TestEnv()
for i in range(10):
    t.reset()
    state = t.get_state()
    print(state)
    a = alg.model.forward(state, True)[-1]
    print(a)
    action = np.argmax(a)
    _, _, _, _ = t.step(action, True)
    print(action)