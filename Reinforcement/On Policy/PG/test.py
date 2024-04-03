# Proper implementation of what was found in PG. Complete version.
# https://explained.ai/matrix-calculus/ use this
# https://www.youtube.com/watch?v=VMj-3S1tku0 WATCH!!!
# Make proper web graph viewing (streamlit?)
# Break down everything into its most basic forms.
# Make the code easy to understand.
# Add proper notation.
import numpy as np
import math
import random
import streamlit
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns


class Layer(object):
    """Object class for one layer of a NN"""
    def __init__(self, nin, nout, activation=''):
        """Initializing all parameters with Xavier initialization"""
        self.w = np.random.randn(nout, nin) / np.sqrt(nin)
        self.b = np.random.randn(nout) / np.sqrt(nout)
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
        p_div = self._activate(p_div, True)
        b_grad = np.squeeze(p_div)
        g_grad = np.vstack(p_div)
        dg = np.dot(g_grad, x.T)
        dn = np.dot(self.w.T, g_grad)
        
        return dg, b_grad, dn
    

class Model(object):
    def __init__(self, config, layer, act):
        if len(act) < len(layer) - 1:
            raise ValueError("Activation length needs to be at least one less than length of layer.")
        
        self.lr = config.get('lr', 0.1)
        self.discount = config.get('discount', 0.99)
        self.decay = config.get('decay', 0.99)
        
        self.layers = []
        
        for i in range(len(layer) - 1):
            self.layers.append(Layer(layer[i], layer[i + 1], act[i]))
        
    def __repr__(self):
        return f"Model: {self.layers}"
    
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
            w_grad, b_grad, prev_grad = layer._backwards(prev_grad, passes[ind])
            
            dw.insert(0, w_grad)
            db.insert(0, b_grad)
        
        return dw, db
            

def mean_squared_error(y_true, y_pred):
    """Calculate mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

X_train = np.random.randn(100,)
y_train = 3 * X_train + 2 + np.random.randn(100,) * 0.1  # y = 3x + 2 + noise

model = Model({}, [100, 400, 100], ['', '', ''])
loss_l = []
epochs = 1000
for epoch in range(epochs):
    passes = model.forward(X_train)
    y_pred = passes.pop(-1) 
    
    loss = mean_squared_error(y_train, y_pred)
    loss_l.append(loss)
    
    grad = 2 * (y_pred - y_train) / len(X_train)
    dw, db = model.backward(passes, grad)
    
    for i, layer in enumerate(model.layers):
        layer.w -= model.lr * dw[i]
        layer.b -= model.lr * db[i]
    
    model.lr *= model.decay

    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss}')
    
plt.plot(np.arange(len(loss_l)), loss_l)
plt.show()


passes = model.forward(X_train)
y_pred = passes[-1]
test_loss = mean_squared_error(y_train, y_pred)
print(f'Test Loss: {test_loss}')