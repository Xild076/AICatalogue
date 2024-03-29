import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import time


class Neuron(object):
    def __init__(self, x, y):
        self.w = np.random.randn(y, x) / np.sqrt(x)


class Activation:
    def relu(x, d=False):
        if d:
            return np.where(x > 0, 1, 0)
        
        return np.maximum(0, x)
    
    def sigmoid(x, d=False):
        if d:
            sig_x = Activation.sigmoid(x)
            return sig_x * (1 - sig_x)
        return 1/(1 + np.exp(-x)) 
    
    def softmax(x, d=False):
        if d:
            return Activation.sigmoid(x, d)
        
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def tanh(x, d=False):
        if d:
            return 1 - np.tanh(x) ** 2
        
        return np.tanh(x)


class Policy(object):
    def __init__(self, config, *model_arg):
        self.lr = config.get('lr', 0.1)
        self.discount = config.get('discount', 0.99)
        self.decay = config.get('decay', 0.99)
        
        self.model = []
        self.activation = []
        for arg in model_arg:
            if isinstance(arg, Neuron):
                self.model.append(arg)
            else:
                self.activation.append(arg)
    
    def forward(self, x):
        passes = []
        hid = x
        for ind, w in enumerate(self.model):
            hid = np.dot(w.w, hid)
            if ind != len(self.model) - 1:
                hid = self.activation[ind](hid)
            passes.append(hid)
        return passes
    
    def fix_hid(self, hid):
        vhidden = []
        for layer in range(len(hid[0])):
            one_layer = []
            for i in range(len(hid)):
                one_layer.append(hid[i][layer])
            vhidden.append(np.array(one_layer))
        return vhidden
    
    def backwards(self, vstate, hid_list, vgrad):
        vhid = self.fix_hid(hid_list)
        dw = []
        dh = vgrad
        
        for ind, hid in enumerate(reversed(vhid)):
            dwl = np.dot(dh.T, hid)
            dw.insert(0, dwl)
            dh = np.dot(dh, self.model[-(ind+1)].w)
            dh[hid <= 0] = 0
        
        dw.insert(0, np.dot(dh.T, vstate))
        
        return dw
    
    def train(self, config):
        epochs = config.get('epochs', 1000)
        batch_size = config.get('batch', 10)
        env = config.get('env')
        reward_list = []
        
        for epoch in range(epochs):
            grad_buffer = [np.zeros_like(neuron.w) for neuron in self.model]
            rmsprop = [np.zeros_like(neuron.w) for neuron in self.model]
            sstate, shidden, sgrads, srewards = [], [], [], []
            state = env.reset()
            old_state = 0
            done = False
            counter = 0
            reward_sum = 0
            
            while not done:
                counter += 1
                calc_state = state - old_state
                passes = self.forward(calc_state)
                prob = passes.pop(-1)
                action, prob = self.sample_action(prob)
                sstate.append(calc_state)
                shidden.append(passes)
                sgrads.append(self.action_grad(action, prob))
                state, reward, done, _ = env.step(action)
                srewards.append(reward)
                reward_sum += reward

            vstate = np.vstack(sstate)
            vgrads = np.vstack(sgrads)
            vrewards = np.vstack(srewards)

            discounted_vrew = self.discount_rewards(vrewards)
            discounted_vrew -= (np.mean(discounted_vrew)).astype(np.float64)
            discounted_vrew /= ((np.std(discounted_vrew)).astype(np.float64) + 1e-8)

            vgrads *= discounted_vrew
            grad = self.backwards(vstate, shidden, vgrads)
            for ind, k in enumerate(self.model):
                grad_buffer[ind] += grad[ind]

            if epoch % batch_size == 0:
                for k, v in enumerate(self.model):
                    g = grad_buffer[k]
                    rmsprop[k] = self.decay * rmsprop[k] + (1 - self.decay) * g**2
                    grad_add = self.lr * g / (np.sqrt(rmsprop[k]) + 1e-8)
                    grad_buffer[k] = np.zeros_like(v)
                    self.model[k].w -= grad_add
            
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


class Categorical(Policy):
    def __init__(self, config, *model_args):
        super().__init__(config, *model_args)
    
    def sample_action(self, x):
        x = Activation.softmax(x)
        if random.random() < 0.05:
            return np.random.randint(0, len(x)), x
        return np.argmax(x), x
    
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
        reward = ((0.5 - abs(action - np.argmax(self.numbers))) * 2) ** 3
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


alg = Categorical({'lr': 0.01}, Neuron(4, 100), Activation.relu, Neuron(100, 4))
alg.train({'epochs': 100000, 'batch': 10, 'env': TestEnv()})

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
