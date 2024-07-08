from engine import value
import random


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin, activ=''):
        self.w = value([random.uniform(-1, 1) for _ in range(nin)])
        self.b = value(0)
        self.activ = activ
    
    def __call__(self, x):
        if isinstance(x, list):
            return self._multi(x)
        return self._single(x)
    
    def _single(self, x):
        act = (self.w * x).sum() + self.b
        if self.activ == 'relu':
            act = act.relu()
        return act
    
    def _multi(self, x):
        act = [self._single(x_s) for x_s in x]
        return act
    
    def parameters(self):
        return [self.w, self.b]

    def __repr__(self):
        return f"{self.activ.upper() if self.activ else 'LINEAR'}Neuron({len(self.w.data)})"


class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
    
    def __call__(self, x):
        if isinstance(x, list):
            return self._multi(x)
        return self._single(x)
    
    def _single(self, x):
        return value.combine([n(x) for n in self.neurons])
    
    def _multi(self, x):
        return [value.combine([n(x_s) for n in self.neurons]) for x_s in x]
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    def __init__(self, layers, activ):
        self.layers = [Layer(layers[i], layers[i + 1], activ=activ[i]) for i in range(len(layers) - 1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        if isinstance(x, list):
            return value.combine(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

