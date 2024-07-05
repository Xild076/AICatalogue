from engine import value
import numpy as np
import random


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()


class Neuron(Module):
    def __init__(self, nin, activ='', init_method='xavier'):
        self.activ = activ
        self.weights = self._initialize_weights(nin, init_method)
        self.bias = value(random.random())

    def _initialize_weights(self, nin, init_method):
        l = 1
        if isinstance(init_method, tuple):
            l = init_method[1]
            init_method = init_method[0]
        if init_method == 'xavier':
            std = np.sqrt(2.0 / (nin + 1)) * l
        elif init_method == 'he':
            std = np.sqrt(2.0 / nin) * l
        elif init_method == 'lecun':
            std = np.sqrt(1.0 / nin) * l
        elif init_method == 'orthogonal':
            a = np.random.randn(nin, 1)
            q, r = np.linalg.qr(a)
            q = q * np.sign(np.diag(r))
            std = q.flatten()[:nin]
            return value(std)
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")
        
        return value(np.random.randn(nin) * std)

    def __call__(self, x):
        act = (x * self.weights).sum(-1) + self.bias
        if self.activ == 'relu':
            act = act.relu()
        elif self.activ == 'softmax':
            act = act.softmax()
        # Add other activations as needed
        return act

    def parameters(self):
        return [self.weights, self.bias]

    def __repr__(self):
        return f"Neuron(weights={self.weights}, bias={self.bias})"


class Layer(Module):
    def __init__(self, nin, nout, activ='', init_method='xavier'):
        self.neurons = [Neuron(nin, activ=activ, init_method=init_method) for _ in range(nout)]

    def __call__(self, x):
        out = [neuron(x) for neuron in self.neurons]
        return value.stack(out).transpose()

    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]

    def __repr__(self):
        return f"Layer(neurons={self.neurons})"


class MLP(Module):
    def __init__(self, layers, activ, init_method='xavier'):
        super().__init__()
        self.layers = [Layer(layers[i], layers[i + 1], activ=activ[i], init_method=init_method) for i in range(len(layers) - 1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        if x.shape[0] == 1:
            return x[0]
        return x

    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]

    def __repr__(self):
        return f"MLP(layers={self.layers})"

