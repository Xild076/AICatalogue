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
    
    def sample_action(self, x):
        # Implement for child class.
        
        return NotImplementedError
    
    def forward(self, x):
        passes = []
        for index, layer in enumerate(self.model):
            x = Nueron.forward_pass(x, layer, self.activation[index])
            passes.append(x)
        return passes

    def backwards(self, action_cache, state_cache, hidden_cache):
        
        
        return NotImplementedError
        

I = 2
H = 3
O = 4

inp = np.random.randn(I)

w1 = np.random.randn(I, H)
w2 = np.random.randn(H, H)
w3 = np.random.randn(H, O)

hid1 = np.dot(inp, w1)
print ("hid1", hid1)

hid2 = np.dot(hid1, w2)
print("hid2", hid2)

act = np.dot(hid2, w3)
print("act", act)

dw3 = np.dot(np.vstack(hid2), np.vstack(act).T)
print("dw3", dw3)

dh2 = np.dot(w3, np.vstack(act))
print("dh2", dh2)

dw2 = np.dot(np.vstack(hid1), dh2.T)
print("dw2", dw2)

dh1 = np.dot(w2, dh2)
print("dh1", dh1)

dw1 = np.dot(np.vstack(inp), dh1.T)
print("dw1", dw1)
print(dw1.shape)