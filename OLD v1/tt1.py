import math
from functools import reduce
from operator import mul

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = self._zero_like(data)  # Grad is a nested list of floats mirroring data
        self._backward = lambda: None
        self._prev = _children
        self._op = _op
    
    def _zero_like(self, data):
        if isinstance(data, list):
            return [self._zero_like(x) for x in data]
        return 0

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out_data = self._elementwise_op(self.data, other.data, lambda x, y: x + y)
        out = Value(out_data, (self, other), '+')

        def _backward():
            self.grad = self._elementwise_op(self.grad, out.grad, lambda g, og: g + og)
            other.grad = self._elementwise_op(other.grad, out.grad, lambda g, og: g + og)
        out._backward = _backward
        
        return out
    
    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out_data = self._elementwise_op(self.data, other.data, lambda x, y: x * y)
        out = Value(out_data, (self, other), '*')

        def _backward():
            self.grad = self._elementwise_op(self.grad, other.data, lambda g, y: g + y * out.grad)
            other.grad = self._elementwise_op(other.grad, self.data, lambda g, x: g + x * out.grad)
        out._backward = _backward
        
        return out
    
    def __pow__(self, other):
        if isinstance(other, Value):
            other = other.data
        out_data = self._elementwise_op(self.data, other, lambda x, p: x ** p)
        out = Value(out_data, (self,), f'**{other}')

        def _backward():
            self.grad = self._elementwise_op(self.grad, other, lambda g, p: g + (p * self.data ** (p - 1)) * out.grad)
        out._backward = _backward

        return out
    
    def __neg__(self):
        return self * -1
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return self + (-other)
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1
    
    def relu(self):
        out_data = self._elementwise_op(self.data, 0, lambda x, _: max(0, x))
        out = Value(out_data, (self,), 'ReLU')

        def _backward():
            self.grad = self._elementwise_op(self.grad, out.data, lambda g, od: g + (od > 0) * out.grad)
        out._backward = _backward

        return out

    def tanh(self):
        out_data = self._elementwise_op(self.data, None, lambda x, _: math.tanh(x))
        out = Value(out_data, (self,), 'Tanh')
        
        def _backward():
            self.grad = self._elementwise_op(self.grad, out.data, lambda g, t: g + (1 - t ** 2) * out.grad)
        out._backward = _backward
        
        return out
    
    def sigmoid(self):
        out_data = self._elementwise_op(self.data, None, lambda x, _: 1 / (1 + math.exp(-x)))
        out = Value(out_data, (self,), 'Sigmoid')

        def _backward():
            self.grad = self._elementwise_op(self.grad, out.data, lambda g, s: g + s * (1 - s) * out.grad)
        out._backward = _backward
        
        return out
    
    def log(self):
        out_data = self._elementwise_op(self.data, None, lambda x, _: math.log(x + 1e-8))
        out = Value(out_data, (self,), 'Log')
        
        def _backward():
            self.grad = self._elementwise_op(self.grad, self.data, lambda g, x: g + 1 / (x + 1e-8) * out.grad)
        out._backward = _backward
        
        return out
    
    def sqrt(self):
        return self ** 0.5
    
    def exp(self):
        out_data = self._elementwise_op(self.data, None, lambda x, _: math.exp(x))
        out = Value(out_data, (self,), 'Exp')

        def _backward():
            self.grad = self._elementwise_op(self.grad, out.data, lambda g, od: g + od * out.grad)
        out._backward = _backward

        return out
    
    def abs(self):
        out_data = self._elementwise_op(self.data, 0, lambda x, _: x if x >= 0 else -x)
        return Value(out_data)

    def __abs__(self):
        out_data = self._elementwise_op(self.data, 0, lambda x, _: x if x >= 0 else -x)
        return Value(out_data)

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"
    
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        
        self.grad = self._zero_like(self.data)  # Initialize the gradient as zero
        self.grad = self._elementwise_op(self.grad, 1, lambda g, _: g + 1)  # Set the gradient of the output to 1
        
        for v in reversed(topo):
            v._backward()

    # Utility methods
    def _elementwise_op(self, a, b, op):
        if isinstance(a, list):
            if b is None:
                return [self._elementwise_op(x, None, op) for x in a]
            return [self._elementwise_op(x, y, op) for x, y in zip(a, self._broadcast(b, len(a)))]
        return op(a, b)
    
    def _broadcast(self, data, size):
        if isinstance(data, list):
            return data
        return [data] * size
    
    def shape(self):
        def _shape(x):
            if isinstance(x, list):
                if len(x) == 0:
                    return (0,)
                return (len(x),) + _shape(x[0])
            return ()
        return _shape(self.data)

# Usage examples
a = Value([[1, 2], [3, 4]])
b = Value([[2, 2], [2, 2]])
c = a * b  # Testing multiplication with nested lists
c.backward()
print(c)  # Should show Value with data as [[2, 4], [6, 8]]
print(a.grad)  # Should show gradients for a as [[2, 2], [2, 2]]
print(b.grad)  # Should show gradients for b as [[1, 2], [3, 4]]
