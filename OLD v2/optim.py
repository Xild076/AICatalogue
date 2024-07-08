from nn import Module, MLP
from engine import value
import math


class Optim(Module):
    def __init__(self, p, lr=0.01, beta=(0.9, 0.99)):
        self.p = p
        self.lr = lr
        self.beta = beta
        self._create_cache()
    
    def _create_cache(self):
        pass
    
    def step(self):
        raise NotImplementedError


class SGD(Optim):
    def __init__(self, p, lr=0.01, beta=(0.9, 0.99)):
        super().__init__(p, lr, beta)
    
    def step(self):
        for param in self.p:
            grad = param._map_zip(lambda x, y: self.lr * x, param.grad, None)
            param.step(grad)