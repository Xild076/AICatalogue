from nn import Module


class Optim(Module):
    def __init__(self, p, lr=0.01, beta=(0.9, 0.99)):
        self.parameters = p
        self.lr = lr
        self.beta = beta
        self._create_cache()
    
    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()
    
    def _create_cache(self):
        pass
    
    def step(self):
        raise NotImplementedError


class SGD(Optim):
    def __init__(self, p, lr=0.01, beta=(0.9, 0.99)):
        super().__init__(p, lr, beta)
    
    def step(self):
        for param in self.parameters:
            param.data -= param.grad * self.lr


class RMSProp(Optim):
    def __init__(self, p, lr=0.01, beta=(0.9, 0.99)):
        super().__init__(p, lr, beta)
        self._create_cache()
    
    def _create_cache(self):
        self._cache = [0] * len(self.parameters)
    
    def step(self):
        for i, param in enumerate(self.parameters):
            self._cache[i] = self._cache[i] * self.beta[0] + (1 - self.beta[0]) * param.grad ** 2
            param.data -= self.lr * param.grad / ((self._cache[i])**0.5 + 1e-8)


class ADAM(Optim):
    def __init__(self, p, lr=0.01, beta=(0.9, 0.99)):
        super().__init__(p, lr, beta)
        self._create_cache()
    
    def _create_cache(self):
        self._cache = [[0] * len(self.parameters) for _ in range(2)]
    
    def step(self):
        for i, param in enumerate(self.parameters):
            self._cache[0][i] = self.beta[0] * self._cache[0][i] + (1 - self.beta[0]) * param.grad
            self._cache[1][i] = self.beta[1] * self._cache[1][i] + (1 - self.beta[1]) * param.grad ** 2
            mt = self._cache[0][i] / (1 - self.beta[0])
            vt = self._cache[1][i] / (1 - self.beta[1])
            param.data -= self.lr * mt / ((vt)**0.5 + 1e-8)
