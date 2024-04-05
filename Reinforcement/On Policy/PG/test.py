import numpy as np
import math
import random
import streamlit
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons


class Layer(object):
    """Object class for one layer of a NN"""
    def __init__(self, nin, nout, activation='', lr=0.01):
        """Initializing all parameters with Xavier initialization"""
        self.w = np.random.randn(nout, nin) / np.sqrt(nin) * lr
        self.b = np.random.randn(nout) / np.sqrt(nout) * lr
        self.actv = activation
    
    def __call__(self, x):
        """Forward pass in NN.
        1. 'act = np.dot(self.w, x) + self.b' multiplies all the weights and the inputs and adds bias.
        2. 'self._activate(act) calls the activation function for calculated values"""
        act = np.dot(self.w, x) + self.b
        return self._activate(act)
    
    def _activate(self, x, d=False):
        """Activates the function. There are three current activation functions.
        1. ReLU, rectified linear unit, adds non-linearity.
        2. Sigmoid: squashes the values between [0, 1] and into a sum of 1.
        3. Tanh: squashes the values between [0, 1]"""
        if self.actv == 'relu':
            if d:
                x[x<=0] = 0
                x[x>0] = 1
                return x
            return np.maximum(0, x)
        
        if self.actv == 'sigmoid':
            if d:
                sig = self._activate(x)
                return sig * (1 - sig)
            return 1 / (1 + np.exp(-x))
        
        if self.actv == 'tanh':
            if d:
                th = self._activate(x)
                return 1 - th**2
            return np.tanh(x)
        return x
    
    def parameters(self):
        """Returns the values of the Layer"""
        return self.w, self.b
    
    def __repr__(self):
        """Returns the description of the Layer"""
        return f"{self.actv.upper()} Layer | ({self.w.shape[1]}, {self.w.shape[0]})"
    
    def _backwards(self, p_div, x):
        """Runs a single backwards pass to calculated its derivative relative to the previous one.
        The equation for the output is: o = act(w * x + b)
        The derivative with respect to b is:
        - do/db = 1 * act'(w * x + b)
        The derivative with respect to w is:
        - do/dw = (x) * act'(w * x + b)
        The derivative with respect to the input (x) is:
        - do/dx = (w) * act'(w * x + b)
        do/dx will be the next x for recursive calculations."""
        
        x = np.vstack(x)
        p_div = self._activate(p_div, True).T
        b_grad = np.mean((p_div))
        g_grad = np.vstack(p_div)
        dg = np.dot(g_grad, x.T)
        dn = np.dot(g_grad.T, self.w)
        
        return dg, b_grad, dn
    

class Model(object):
    def __init__(self, config, layer, act):
        if len(act) < len(layer) - 1:
            raise ValueError("Activation length needs to be at least one less than length of layer.")
        
        self.lr = config.get('lr', 0.1)
        self.discount = config.get('discount', 0.99)
        self.decay = config.get('decay', 0.99)
        
        self.beta = config.get('beta', [0.99, 0.99])
        
        self.optm = config.get('optm', 'sgd')
        self.v = config.get('type', 'loss')
        if self.v == 'loss':
            self.lt = config.get('loss', 'mse')
        if self.v == 'reward':
            self.cont = config.get('cont', True)
        
        self.layers = []
        for i in range(len(layer) - 1):
            self.layers.append(Layer(layer[i], layer[i + 1], act[i], self.lr))        
        
    def __repr__(self):
        return f"Model: {self.layers}"
    
    def _set_cache(self):
        self.cache_w = []
        self.cache_b = []
        for l in self.layers:
            self.cache_w.append(np.zeros_like(l.w))
            self.cache_b.append(np.zeros_like(l.b))
    
    def _set_buffer(self):
        self._gb_w = []
        self._gb_b = []
        for l in self.layers:
            self._gb_w.append(np.zeros_like(l.w))
            self._gb_b.append(np.zeros_like(l.b))
    
    def forward(self, x):
        passes = []
        passes.append(x)
        for layer in self.layers:
            x = layer(x)
            passes.append(x)
        return passes
    
    def backward(self, passes, grad):
        passes.reverse()
        
        prev_grad = grad
        
        dw = []
        db = []
        
        for ind, layer in enumerate(reversed(self.layers)):
            w_grad, b_grad, prev_grad = layer._backwards(prev_grad, passes[ind].T)
            
            dw.insert(0, w_grad)
            db.insert(0, b_grad)
        
        return dw, db

    def _loss(self, label, pred, d=False):
        if self.lt == 'mse':
            if d:
                return 2 * (label - pred) / len(label)
            return np.mean((label - pred) ** 2)
        
        if self.lt == 'mae':
            if d:
                return -1 * np.where(label - pred < 0, -1, np.where(label - pred == 0, 0, 1))
            return np.sum(np.abs(label - pred))
        
        if d:
            return 1
        return label - pred
    
    def _reward(self, actions, reward):
        drw = self._discount_rewards(reward)
        drw -= np.mean(drw)
        drw /= (np.std(drw) + 1e-8)
        
        return actions * drw
    
    def _discount_rewards(self, rewards):
        discounted_reward = np.zeros_like(rewards, dtype=np.float64)
        total_rewards = 0
        for t in reversed(range(0, len(rewards))):
            total_rewards = total_rewards * self.discount + rewards[t]
            discounted_reward[t] = total_rewards
        return discounted_reward
    
    def _optimize(self, dw, db):
        if self.optm == 'rmsprop':
            dw_u = []
            db_u = []
            for i in range(len(self.layers)):
                g_w = dw[i]
                self.cache_w[i] = self.cache_w[i] * self.beta[0] + (1 - self.beta[0]) * g_w ** 2
                dw_u.append(self.lr * g_w / np.sqrt(self.cache_w[i] + 1e-8))
                
                g_b = db[i]
                self.cache_b[i] = self.cache_b[i] * self.beta[0] + (1 - self.beta[0]) * g_b ** 2
                db_u.append(self.lr * g_b / np.sqrt(self.cache_b[i] + 1e-8))
            
            return dw_u, db_u
        
        for i in range(len(dw)):
            dw[i] = dw[i] * self.lr
            db[i] = db[i] * self.lr
        return dw, db

    def _fix_hid(self, hid):
        vhidden = []
        for layer in range(len(hid[0])):
            one_layer = []
            for i in range(len(hid)):
                one_layer.append(hid[i][layer])
            vhidden.append(np.array(one_layer))
        return vhidden
    
    def train(self, config):
        self._batch_size = config.get('batch', 10)
        _epoch = config.get('epochs', 1000)
        env = config.get('env', None)
        
        self._set_buffer()
        self._set_cache()
        
        o_list = []
        
        for epoch in range(_epoch):
            o = None
            if self.v == 'loss':
                o = self._epoch_loss(epoch, env)
            if self.v == 'reward':
                o = self._epoch_reward(epoch, env)
            print(epoch, o)
            o_list.append(o)
        
        return o_list
    
    def _epoch_reward(self, epoch, env):
        shidden, sgrads, srewards = [], [], []
        state = env.reset()
        old_state = 0
        reward_sum = 0
        done = False
        
        while not done:
            calc_state = state - old_state
            
            if self.cont:
                old_state = state
            
            passes = self.forward(calc_state)
            act, act_grad = self.sample_action(passes.pop(-1))
            
            shidden.append(passes)
            sgrads.append(act_grad)
            
            state, reward, done, _ = env.step(act)
            
            srewards.append(reward)
            
            reward_sum += reward
        
        vgrads = np.array(sgrads)
        vrewards = np.vstack(srewards)
        
        vgrads = self._reward(vgrads, vrewards)
        g_w, g_b = self.backward(self._fix_hid(shidden), vgrads)
        for i in range(len(self._gb_w)):
            self._gb_w[i] -= g_w[i]
            self._gb_b[i] -= g_b[i]
        
        if epoch % self._batch_size == 0:
            dw, db = self._optimize(self._gb_w, self._gb_b)
            for i in range(len(self.layers)):
                self.layers[i].w -= dw[i]
                self.layers[i].b -= db[i]
            self._set_buffer()
        
        return reward_sum
    
    def _epoch_loss(self, epoch, env):
        x_train, y_train = env.get_train()
        
        passes = self.forward(x_train)
        y_pred = passes.pop(-1)
        
        loss = self._loss(y_train, y_pred)
        grad = self._loss(y_train, y_pred, True)
        
        g_w, g_b = self.backward(passes, grad)

        for i in range(len(self._gb_w)):
            self._gb_w[i] += g_w[i]
            self._gb_b[i] += g_b[i]
        
        if epoch % self._batch_size == 0:
            dw, db = self._optimize(self._gb_w, self._gb_b)
            for i in range(len(self.layers)):
                self.layers[i].w += dw[i]
                self.layers[i].b += db[i]
            self._set_buffer()
        
        return loss
    
    def sample_action(self, x):
        """Return an action and action_grad"""
        exp_x = np.exp(x - np.max(x))
        prob = exp_x / np.sum(exp_x)
        if random.random() < 0.05:
            act = random.randint(0, len(prob) - 1)
        else:
            act = np.argmax(prob)
        aoh = np.zeros_like(prob)
        aoh[act] = 1
        
        return act, aoh - prob


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


m = Model({'lr': 0.00001, 'optm': 'rmsprop', 'type': 'loss', 'cont': False}, [100, 200, 100], ['', '', ''])
j = m.train({'epochs': 10000, 'batch': 10, 'env': TestEnvironment()})

sns.lineplot(j)
plt.show()

m = Model({'lr': 0.001, 'optm': 'rmsprop', 'type': 'reward', 'cont': False}, [3, 100, 3], ['', '', ''])
l = m.train({'epochs': 10000, 'batch': 25, 'env': TestEnv()})

sns.lineplot(l)
plt.show()