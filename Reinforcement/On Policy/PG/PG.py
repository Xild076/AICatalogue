import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns


class Nueron(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.shape = {
            'weight': tuple([x, y]),
            'bias': y
        }
        self.weight = np.random.randn(x, y)
    
    def __str__(self):
        return f"Weight: {self.weight}"
    
    def update(self, add):
        self.weight += add
    
    @staticmethod
    def forward_pass(x, y, activation):
        return activation.activation(np.dot(x, y.weight), activation.alpha)


class Activation(object):
    def __init__(self, activation, alpha=0.99):
        self.activation = activation
        self.alpha = alpha
    
    @staticmethod
    def relu(x, alpha, derivative=False):
        if derivative:
            return np.where(x > 0, 1, 0)
        
        return np.maximum(0, x)
    
    @staticmethod
    def sigmoid(x, alpha, derivative=False):
        if derivative:
            sig_x = 1/(1 + np.exp(-x))
            return sig_x * (1 - sig_x)
        return 1/(1 + np.exp(-x))
    
    @staticmethod
    def softmax(x, alpha, derivative=False):
        if derivative:
            return Activation.sigmoid(x, alpha, derivative)
        
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    @staticmethod
    def tanh(x, alpha, derivative=False):
        if derivative:
            return 1 - np.tanh(x) ** 2
        
        return np.tanh(x)
    
    @staticmethod
    def prelu(x, alpha, derivative=False):
        if derivative:
            np.where(x > 0, 1, alpha)
        
        return np.where(x > 0, 1, alpha)
    
    @staticmethod        
    def leaky_relu(x, alpha, derivative=False):
        if derivative:
            np.where(x > 0, 1, alpha)
        
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def elu(x, alpha, derivative=False):
        if derivative:
            return np.where(x > 0, 1, alpha * np.exp(x))
        
        return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

    @staticmethod
    def siwsh(x, alpha, derivative=False):
        if derivative:
            swish_x = x / (1 + np.exp(-x))
            sigmoid_x = Activation.sigmoid(x, alpha)
            return swish_x + sigmoid_x * (1 - swish_x)
        
        return x / (1 + np.exp(-x))


"""class Optimizer(object):
    def __init__(self, Policy, optimizer, beta):
        Optimizer.set_values()
        
        self.model = Policy.model
        self.lr = Policy.learning_rate
        self.optimizer = optimizer
        self.cache_size = self.optimizer.cache_size
        self.cache = [[np.zeros_like(neuron.weight) for neuron in self.model] * self.cache_size]
        self.beta = beta
        if len(self.beta) != self.optimizer.beta_size:
            raise ValueError("Wrong beta amount.")
    
    def optimize(self, grads_buffer, epoch):
        for index, grad in enumerate(grads_buffer):
            grad_add, rc = self.optimizer(grad, self.cache)

    
    @staticmethod
    def set_values():
        Optimizer.rmsprop.cache_size = 1
        Optimizer.rmsprop.beta_size = 1
    
    @staticmethod
    def rmsprop(grad, cache, beta, lr, epoch):
        rc = beta[0] * cache[0] + (1 - beta[0]) * grad**2
        
        Optimizer.rmsprop.cache_size = 1
        Optimizer.rmsprop.beta_size = 1
        
        caches = [rc]
        
        return (lr * grad / (np.sqrt(rc) + 1e-8)), caches
"""


class Policy(object):
    def __init__(self, *args):       
        self.model = []
        for arg in args:
            if isinstance(arg, Nueron):
                self.model.append(arg)
        
        self.activation = []
        for arg in args:
            if isinstance(arg, Activation):
                self.activation.append(arg)
    
    def config(self, config):
        self.learning_rate = config.get('learning_rate', 0.01)
    
    def sample_action(self, action_prob):
        return np.random.choice(len(action_prob), p=action_prob)
    
    def action_grad(self, action, action_prob):
        aoh = np.zeros(len(action_prob))
        aoh[action] = 1
        return aoh - action_prob

    def discount_rewards(self, rewards):
        discounted_reward = np.zeros_like(rewards, dtype=np.float64)
        total_rewards = 0
        for t in reversed(range(0, len(rewards))):
            total_rewards = total_rewards * 0.99 + rewards[t]
            discounted_reward[t] = total_rewards
        return discounted_reward
    
    def train(self, config):
        epochs = config.get('epochs', 1000)
        batch_size = config.get('batch', 10)
        env = config.get('env')
        
        reward_list = []
        
        for epoch in range(epochs):

            grad_buffer = [np.zeros_like(neuron.weight) for neuron in self.model]

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
            grad = self.backward(vstate.T, vhidden, vgrads.T)
            grad_buffer = [gb + g for gb, g in zip(grad_buffer, grad)]

            if epoch % batch_size == 0:
                for i in range(len(self.model)):
                    g = grad_buffer[i]
                    grad_add = g * self.learning_rate
                    grad_buffer[i] = np.zeros_like(self.model[i].weight)
                    self.model[i].update(grad_add)

            reward_list.append(reward_sum)
            print(epoch, reward_sum)
        
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=range(len(reward_list)), y=reward_list, color='blue', label='Rewards')

        average_growth = sum(reward_list) / len(reward_list)

        plt.axhline(average_growth, color='red', linestyle='--', label='Average Growth')

        plt.xlabel('Time')
        plt.ylabel('Reward')
        plt.title('Reward Growth Over Time')
        plt.legend()

        plt.grid(True)
        plt.show()

        return reward_list
                
    def backward(self, vstate, vhid, vgrad):   
        dh = vgrad
        update_weight = []
        for i in range(len(vhid[0])):
            objects = np.vstack([item[-(i+1)] for item in vhid])
            dw = np.dot(dh, objects).T
            update_weight.append(dw)
            dh = np.dot(self.model[-(i+1)].weight, dh)
        update_weight.append(np.dot(dh, vstate.T))
        return update_weight
        
    
    def forward(self, x):
        passes = []
        for index, layer in enumerate(self.model):
            x = Nueron.forward_pass(x, layer, self.activation[index])
            passes.append(x)
        return passes

class Categorical(Policy):
    def __init__(self, *args):
        super().__init__(*args)
    
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
        self.numbers = [random.random(), random.random(), random.random()]
        self.counter = 0
    
    def reset(self):
        self.numbers = [random.random(), random.random(), random.random()]
        self.counter = 0
        return self.get_state()
            
    def get_state(self):
        self.numbers = [random.random(), random.random(), random.random()]
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


alg = Categorical(Nueron(3, 20), Activation(Activation.relu), Nueron(20, 20), Activation(Activation.relu), Nueron(20, 3), Activation(Activation.softmax))
alg.config({})
alg.train({'epochs': 10000, 'batch': 10, 'env': TestEnv()})