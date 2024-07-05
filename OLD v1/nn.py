from engine import Value, Array
import random


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin, activ=''):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.activ = activ

    def __call__(self, x):
        act = sum(x * Array(self.w)) + self.b
        if self.activ == 'relu':
            act = act.relu()
        elif self.activ == 'sigmoid':
            act = act.sigmoid()
        elif self.activ == 'tanh':
            act = act.tanh()
        return act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{self.activ.upper() if self.activ else 'LINEAR'}Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = Array(Neuron(nin, **kwargs) for _ in range(nout))

    def __call__(self, x):
        out = Array(n(x) for n in self.neurons)
        return out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    def __init__(self, layers, activ):
        self.layers = Array(Layer(layers[i], layers[i + 1], activ=activ[i]) for i in range(len(layers) - 1))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
    
    