from nn import Module
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
            param.data -= param.grad * self.lr


class RMSProp(Optim):
    def __init__(self, p, lr=0.01, beta=(0.9, 0.99)):
        super().__init__(p, lr, beta)
        self._create_cache()
    
    def _create_cache(self):
        self._cache = [0] * len(self.p)
    
    def step(self):
        for i, param in enumerate(self.p):
            self._cache[i] = self._cache[i] * self.beta[0] + (1 - self.beta[0]) * param.grad ** 2
            param.data -= self.lr * param.grad / ((self._cache[i])**0.5 + 1e-8)


class ADAM(Optim):
    def __init__(self, p, lr=0.01, beta=(0.9, 0.99)):
        super().__init__(p, lr, beta)
        self._create_cache()
    
    def _create_cache(self):
        self._cache = [[0] * len(self.p) for _ in range(2)]
    
    def step(self):
        for i, param in enumerate(self.p):
            self._cache[0][i] = self.beta[0] * self._cache[0][i] + (1 - self.beta[0]) * param.grad
            self._cache[1][i] = self.beta[1] * self._cache[1][i] + (1 - self.beta[1]) * param.grad ** 2
            mt = self._cache[0][i] / (1 - self.beta[0])
            vt = self._cache[1][i] / (1 - self.beta[1])
            param.data -= self.lr * mt / ((vt)**0.5 + 1e-8)
