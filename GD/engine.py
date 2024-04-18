import math
import random
import numpy as np
import matplotlib.pyplot as plt


class Galue(object):
    def __init__(self, value, _children=()):
        self.value = value
        self.grad = 0
        
        self._backward = lambda: None
        self._prev = set(_children)
    
    def __repr__(self) -> str:
        return f"Galue (value={self.value}, grad={self.grad})"
    
    def __add__(self, other):
        other = other if isinstance(other, Galue) else Galue(other)
        output = Galue(self.value + other.value, (self, other))
        
        def _backward():
            self.grad += output.grad
            other.grad += output.grad
        output._backward = _backward
        
        return output
    
    def __mul__(self, other):
        other = other if isinstance(other, Galue) else Galue(other)
        output = Galue(self.value * other.value, (self, other))
        
        def _backward():
            self.grad += output.grad * other.value
            other.grad += output.grad * self.value
        output._backward = _backward
        
        return output
    
    def __pow__(self, other):
        other = other if isinstance(other, Galue) else Galue(other)
        output = Galue(self.value ** other.value, (self, other))
        
        def _backward():
            self.grad += output.grad * other.value * (self.value ** (other.value - 1))
        output._backward = _backward
        
        return output
    
    def backward(self, new=True):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        if new:
            self.grad = 1
        
        for v in reversed(topo):
            v._backward()
    
    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1
    
    def relu(self):
        output = Galue(0 if self.value < 0 else self.value, (self,))
        
        def _backward():
            self.grad += (output.value > 0) * output.grad
        output._backward = _backward
        
        return output
    
    def tanh(self):
        output = Galue(math.tanh(self.value), (self,))
        
        def _backward():
            self.grad += (1 - math.tanh(output.value) ** 2) * output.grad
        output._backward = _backward
        
        return output
    
    def sigmoid(self):
        output = Galue((1 / (1 + math.exp(-self.value))), (self,))
        
        def _backward():
            e_x = (1 / (1 + math.exp(-output.value)))
            self.grad += (e_x * (1 - e_x)) * output.grad
        output._backward = _backward
        
        return output
    
    def log(self):
        output = Galue(math.log(self.value), (self,))
        
        def _backward():
            self.grad += 1 / self.value * output.grad 
        output._backward = _backward
        
        return output
    
    def sqrt(self):
        output = Galue(math.sqrt(self.value), (self,))
        
        def _backward():
            self.grad += output.grad * 0.5 / (math.sqrt(self.value))
        output._backward = _backward
        
        return output
    
    def __abs__(self):
        if self.value < 0:
            return self.value * -1
        return self
    
    @staticmethod
    def softmax(galues):
        max_value = max(galues, key=lambda x: x.value).value
        exp_values = [math.exp(g.value - max_value) for g in galues]
        sum_exp_values = sum(exp_values)
        softmax_values = Garray()
        for ind, exp_val in enumerate(exp_values):
            softmax_values.append(Galue(exp_val / sum_exp_values, (galues[ind],)))
        
        def _backward():
            for g, softmax_val in zip(galues, softmax_values):
                g.grad += softmax_val.grad * (softmax_val.value * (1 - softmax_val.value))
        for value in softmax_values:
            value._backward = _backward
        
        return softmax_values

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, size, activ=''):
        self.w = [Galue(random.uniform(-1,1)) for _ in range(size)]
        self.b = Galue(random.uniform(-1,1))
        self._activ = activ
    
    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self._activ == 'relu':
            act.relu()
        elif self._activ == 'tanh':
            act.tanh()
        elif self._activ == 'sigmoid':
            act.sigmoid()
        
        return act

    def parameters(self):
        return self.w + [self.b]
    
    def __repr__(self):
        return f"{self._activ.upper()} Neuron ({len(self.w)})"
    

class Layer(Module):
    def __init__(self, v_in, v_out, activ=''):
        self.neurons = [Neuron(v_in, activ) for _ in range(v_out)]
    
    def __call__(self, x):
        output = [n(x) for n in self.neurons]
        return output
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self):
        return f"Layer [{', '.join(str(n) for n in self.neurons)}]"


class Garray(list):
    def __init__(self, *args):
        super().__init__(args)
        self._normalize()
    
    def _normalize(self):
        self[:] = [Galue(item) if not isinstance(item, Galue) and not isinstance(item, list) and not isinstance(item, Garray) else item for item in self]
    
    def _apply_function(self, func):
        return Garray(*(func(item) for item in self))

    def relu(self):
        return self._apply_function(Galue.relu)

    def tanh(self):
        return self._apply_function(Galue.tanh)

    def sigmoid(self):
        return self._apply_function(Galue.sigmoid)

    def log(self):
        return self._apply_function(Galue.log)
    
    def sqrt(self):
        return self._apply_function(Galue.sqrt)

    def softmax(self):
        return Galue.softmax(self)
    
    def backward(self):
        for item in self:
            item.backward()
    
    def __repr__(self) -> str:
        return f'Garray {[item for item in self]}'

    @staticmethod
    def garrayify(l: list):
        return Garray(*l)
    
    @staticmethod
    def neg(a):
        return Garray(*(-item for item in a))
    
    @staticmethod
    def add(a, b):
        if isinstance(a, (Garray, list)) and isinstance(b, (int, float, Galue)):
            return Garray(*(a[i] + b for i in range(len(a))))
        if isinstance(b, (Garray, list)) and isinstance(a, (int, float, Galue)):
            return Garray(*(a + b[i] for i in range(len(b))))
        if isinstance(a, (list, Garray)) and isinstance(a, (list, Garray)) and len(a) == len(b):
            return Garray(*(a[i] + b[i] for i in range(len(a))))
        raise ValueError("Something went wrong.")
    
    @staticmethod
    def sub(a, b):
        if isinstance(a, (Garray, list)) and isinstance(b, (float, int, Galue)):
            return Garray(*(a[i] - b for i in range(len(a))))
        if isinstance(b, (Garray, list)) and isinstance(a, (float, int, Galue)):
            return Garray(*(a - b[i] for i in range(len(b))))
        if isinstance(a, (list, Garray)) and isinstance(a, (list, Garray)) and len(a) == len(b):
            return Garray(*(a[i] - b[i] for i in range(len(a))))
        raise ValueError("Something went wrong.")
    
    @staticmethod
    def mul(a, b):
        if isinstance(a, (Garray, list)) and isinstance(b, (float, int, Galue)):
            return Garray(*(a[i] * b for i in range(len(a))))
        if isinstance(b, (Garray, list)) and isinstance(a, (float, int, Galue)):
            return Garray(*(a * b[i] for i in range(len(b))))
        if isinstance(a, (list, Garray)) and isinstance(a, (list, Garray)) and len(a) == len(b):
            return Garray(*(a[i] * b[i] for i in range(len(a))))
        raise ValueError("Something went wrong.")
    
    @staticmethod
    def div(a, b):
        if isinstance(a, (Garray, list)) and isinstance(b, (float, int, Galue)):
            return Garray(*(a[i] / b for i in range(len(a))))
        if isinstance(b, (Garray, list)) and isinstance(a, (float, int, Galue)):
            return Garray(*(a / b[i] for i in range(len(b))))
        if isinstance(a, (list, Garray)) and isinstance(a, (list, Garray)) and len(a) == len(b):
            return Garray(*(a[i] / b[i] for i in range(len(a))))
        raise ValueError("Something went wrong.")
    
    @staticmethod
    def pow(a, b):
        if isinstance(a, (Garray, list)) and isinstance(b, (float, int, Galue)):
            return Garray(*(a[i] ** b for i in range(len(a))))
        if isinstance(b, (Garray, list)) and isinstance(a, (float, int, Galue)):
            return Garray(*(a ** b[i] for i in range(len(b))))
        if isinstance(a, (list, Garray)) and isinstance(a, (list, Garray)) and len(a) == len(b):
            return Garray(*(a[i] ** b[i] for i in range(len(a))))
        raise ValueError("Something went wrong.")
    
    @staticmethod
    def abs(a):
        return Garray(*(abs(item) for item in a))
    
    @staticmethod
    def min(a):
        if not isinstance(a, (list, Garray)):
            raise ValueError("Input must be a list or Garray.")
        if len(a) == 0:
            raise ValueError("Input list or Garray must not be empty.")
        
        if isinstance(a, list):
            a = Garray.garrayify(a)
        
        min_val = a[0]
        for item in a[1:]:
            min_val = item if item.value < min_val.value else min_val
        return min_val
    
    @staticmethod
    def max(a):
        if not isinstance(a, (list, Garray)):
            raise ValueError("Input must be a list or Garray.")
        if len(a) == 0:
            raise ValueError("Input list or Garray must not be empty.")
        
        if isinstance(a, list):
            a = Garray.garrayify(a)
        
        max_val = a[0]
        for item in a[1:]:
            max_val = item if item.value > max_val.value else max_val
        return max_val


class GD(Module):
    def __init__(self, layers, activ):
        self.layers = [Layer(layers[i], layers[i+1], activ[i]) for i in range(len(layers) - 1)]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return Garray.garrayify(x)

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"GD of [{', '.join(str(layer) for layer in self.layers)}]"


class Loss:
    """Types: 
    Mean Absolute Error (MAE) Loss
    Mean Squared Error (MSE) Loss
    Mean Bias Error (MBE) Loss
    Categorical Cross Entropy (CCE) Loss
    Binary Cross Entropy (BCE) Loss
    Huber Loss
    Hinge Loss
    Cross Entropy Regression (CER) Loss
    KL Divergence (KLD) Loss
    Smooth L1 (SL1) Loss
    Poisson Loss
    Quantile Loss
    Wasserstein Loss
    Triplet Loss
    """
    
    @staticmethod
    def mean_absolute_error(pred, real):
        return sum(Garray.abs(Garray.sub(pred, real))) / len(pred)
    
    @staticmethod
    def mean_squared_error(pred, real):
        return sum(Garray.pow(Garray.sub(pred, real), 2)) / len(pred)
    
    @staticmethod
    def mean_bias_error(pred, real):
        return sum(Garray.sub(pred, real)) / len(pred)
    
    @staticmethod
    def categorical_cross_entropy(pred, real):
        return -sum(Garray.mul(real, pred.log())) / len(pred)
    
    @staticmethod
    def binary_cross_entropy(pred, real):
        term_0 = Garray.mul(Garray.sub(1.0, real), Garray.sub(1 + 1e-8, pred).log())
        term_1 = Garray.mul(real, pred.log())
        return -sum(Garray.add(term_0, term_1)) / len(pred)
    
    @staticmethod
    def huber(pred, real, delta=1.0):
        absolute_errors = Garray.abs(Garray.sub(pred, real))
        min_errors = Garray.min(absolute_errors)
        huber_term = 0.5 * Garray.min([min_errors, delta]) ** 2
        linear_term = Garray.mul(delta, Garray.sub(absolute_errors, Garray.min([min_errors, delta])))
        return sum(Garray.add(huber_term, linear_term)) / len(pred)
    
    @staticmethod
    def hinge(pred, real):
        return sum(Garray.relu(Garray.sub(1, Garray.mul(pred, real)))) / len(pred)
    
    @staticmethod
    def cross_entropy_regression(pred, real):
        return -sum(Garray.mul(real, pred.log())) / len(pred)
    
    @staticmethod
    def kl_divergence(pred, target):
        if not isinstance(target, Garray):
            target = Garray.garrayify(target)
            target = Garray.add(target, 1e-8)
        return sum(Garray.mul(target, Garray.sub(target.log(), pred.log()))) / len(pred)
    
    @staticmethod
    def smooth_l1(pred, real, beta=1.0):
        absolute_errors = Garray.abs(Garray.sub(pred, real))
        smooth_l1_term = Garray.mul(0.5 / beta ** 2, Garray.pow(Garray.add(beta ** 2, absolute_errors), 0.5))
        return sum(smooth_l1_term) / len(pred)
    
    @staticmethod
    def poisson(pred, real):
        return sum(Garray.sub(pred, Garray.mul(real, Garray.log(pred)))) / len(pred)
    
    @staticmethod
    def quantile(pred, real, quantile=0.5):
        if not isinstance(quantile, float) or quantile < 0 or quantile > 1:
            raise ValueError("Quantile must be a float between 0 and 1.")
        error = Garray.sub(real, pred)
        loss = Garray.mul(quantile, Garray.relu(error)) + Garray.mul(1 - quantile, Garray.relu(Garray.neg(error)))
        return sum(loss) / len(pred)
    
    @staticmethod
    def wasserstein(pred, real):
        return sum(Garray.abs(Garray.sub(real, pred))) / len(pred)
    
    @staticmethod
    def triplet(anchor, positive, negative, margin=1.0):
        dist_positive = sum(Garray.pow(Garray.sub(anchor, positive), 2))
        dist_negative = sum(Garray.pow(Garray.sub(anchor, negative), 2))
        return Garray.relu(dist_positive - dist_negative + margin)


class Optim(Module):
    def __init__(self, optmtype, parameters, lr=0.01, beta=[0.9, 0.99]):
        self.type = optmtype
        self.parameters = parameters
        self.lr = lr
        self.beta = beta
        self._create_cache()
    
    def _create_cache(self):
        if self.type == 'rmsprop':
            self._cache = [0 for _ in self.parameters]
        if self.type == 'adam':
            self._cache = [[0 for _ in self.parameters] for i in range(2)]
    
    def optimize(self):
        if self.type == 'sgd':
            for param in self.parameters:
                param.value -= param.grad * self.lr
        elif self.type == 'rmsprop':
            for i, param in enumerate(self.parameters):
                self._cache[i] = self._cache[i] * self.beta[0] + (1 - self.beta[0]) * param.grad ** 2
                param.value -= self.lr * param.grad / np.sqrt(self._cache[i] + 1e-8)
        elif self.type == 'adam':
            for i, param in enumerate(self.parameters):
                self._cache[0][i] = self.beta[0] * self._cache[0][i] + (1 - self.beta[0]) * param.grad
                self._cache[1][i] = self.beta[1] * self._cache[1][i] + (1 - self.beta[1]) * param.grad ** 2
                mt = self._cache[0][i] / (1 - self.beta[0])
                vt = self._cache[1][i] / (1 - self.beta[1])
                param.value -= self.lr * mt / (math.sqrt(vt) + 1e-8)
        else:
            raise ValueError("Optimization doesn't exist.")


model = GD([3, 2, 1], ['relu', ''])
optim = Optim('adam', model.parameters(), 0.003)
a = []
for i in range(1000):
    g = model([3, 2, 1])
    l = Loss.mean_squared_error(g, [2])
    print("l", l.value)
    a.append(l.value)
    l.backward()
    optim.optimize()
    model.zero_grad()

plt.plot(a)
plt.show()