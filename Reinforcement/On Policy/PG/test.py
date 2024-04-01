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
        return f"{self.actv.upper()} Layer | {self.w.shape}"
    
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