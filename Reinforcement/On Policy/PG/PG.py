import numpy as np


class Nueron(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.shape = {
            'weight': tuple([x, y]),
            'bias': y
        }
        self.weight = np.random.randn(x, y)
        self.bias = np.random.randn(y)
    
    def __str__(self):
        return f"Weight: {self.weight}, Bias: {self.bias}"
    
    @staticmethod
    def forward_pass(x, y, activation):
        return activation.activation(np.dot(x, y.weight) + y.bias, activation.alpha)


class Activation(object):
    def __init__(self, activation, alpha=0.99):
        self.activation = activation
        self.alpha = alpha
    
    @staticmethod
    def relu(x, alpha):
        return np.maximum(0, x)
    
    @staticmethod
    def sigmoid(x, alpha):
        return 1/(1 + np.exp(-x))
    
    @staticmethod
    def softmax(x, alpha):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    @staticmethod
    def tanh(x, alpha):
        return np.tanh(x)
    
    @staticmethod
    def prelu(x, alpha):
        return np.where(x > 0, 1, alpha)
    
    @staticmethod        
    def leaky_relu(x, alpha):
            return np.where(x > 0, x, alpha * x)

    @staticmethod
    def elu(x, alpha):
        return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

    @staticmethod
    def siwsh(x, alpha):
        return x / (1 + np.exp(-x))
    
    def derivatives():
        return NotImplementedError


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
    
    def forward(self, x):
        passes = []
        for index, layer in enumerate(self.model):
            x = Nueron.forward_pass(x, layer, self.activation[index])
            passes.append(x)
        return passes

    def backwards(self, action_cache, state_cache, hidden_cache):
        return NotImplementedError
        

I = 20
H = 24
O = 10


model = Model(Nueron(I, H), Activation(Activation.relu), Nueron(H, H), Activation(Activation.sigmoid), Nueron(H, O), Activation(Activation.siwsh))

inp = np.random.randn(I)

print(model.forward(inp))