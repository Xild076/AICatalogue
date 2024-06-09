import math
import math
import time
import types

def is_single(item):
    return isinstance(item, (int, float, Value))

def is_multiple(item):
    return isinstance(item, (list, set, tuple))


class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = _children
        self._op = _op
    
    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        
        return out
    
    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out
    
    def __pow__(self, other):
        if isinstance(other, Value):
            other = other.data
        out = Value(self.data ** other, (self,), f'**{other}')
        
        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
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
        out_data = 0 if self.data < 0 else self.data
        out = Value(out_data, (self,), 'ReLU')
        
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'Tanh')
        
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        
        return out
    
    def sigmoid(self):
        s = 1 / (1 + math.exp(-self.data))
        out = Value(s, (self,), 'Sigmoid')
        
        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        
        return out
    
    def log(self):
        out_data = math.log(self.data + 1e-8)
        out = Value(out_data, (self,), 'Log')
        
        def _backward():
            self.grad += 1 / (self.data + 1e-8) * out.grad
        out._backward = _backward
        
        return out
    
    def sqrt(self):
        return self ** 0.5
    
    def exp(self):
        out_data = math.exp(self.data)
        out = Value(out_data, (self,), 'Exp')

        def _backward():
            self.grad += out_data * out.grad
        out._backward = _backward

        return out
    
    def abs(self):
        if self.data < 0:
            return -self
        return self
    
    def __abs__(self):
        if self.data < 0:
            return self * -1
        return self
    
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
        
        self.grad = 1
        
        for v in reversed(topo):
            v._backward()


class Array(list):
    def __init__(self, *args):
        if args and isinstance(args[0], list):
            super().__init__(args[0])
        elif isinstance(args[0], types.GeneratorType):
            super().__init__(*args)
        else:
            super().__init__(args)
        
        self._normalize()
    
    def _normalize(self):
        for i in range(len(self)):
            if isinstance(self[i], (int, float)):
                self[i] = Value(self[i])
    
    def __add__(self, other):
        if is_single(other):
            return Array(a + other for a in self)
        elif is_multiple(other):
            if len(self) == len(other):
                return Array(a + b for a, b in zip(self, other)) 
            else:
                raise ValueError(f"Other is len {len(other)} and self is len {len(self)}. Please input arrays of the same size.")
        else:
            raise ValueError(f"Other is type {type(other)}. Please use inputs of supported type.")
    
    def __mul__(self, other):
        if is_single(other):
            return Array(a * other for a in self)
        elif is_multiple(other):
            if len(self) == len(other):
                return Array(a * b for a, b in zip(self, Array(other))) 
            else:
                raise ValueError(f"Other is len {len(other)} and self is len {len(self)}. Please input arrays of the same size.")
        else:
            raise ValueError(f"Other is type {type(other)}. Please use inputs of supported type.")
    
    def __pow__(self, other):
        if is_single(other):
            return Array(a ** other for a in self)
        elif is_multiple(other):
            if len(self) == len(other):
                return Array(a ** b for a, b in zip(self, other)) 
            else:
                raise ValueError(f"Other is len {len(other)} and self is len {len(self)}. Please input arrays of the same size.")
        else:
            raise ValueError(f"Other is type {type(other)}. Please use inputs of supported type.")
    
    def __neg__(self):
        return -1 * self
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return self + (-other)
    
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __rtruediv__(self, other):
        return self * (other ** -1)
    
    def _apply_function(self, func):
        return Array(func(item) for item in self)
    
    def relu(self):
        return self._apply_function(Value.relu)
    
    def tanh(self):
        return self._apply_function(Value.tanh)
    
    def sigmoid(self):
        return self._apply_function(Value.sigmoid)

    def log(self):
        return self._apply_function(Value.log)
    
    def exp(self):
        return self._apply_function(Value.exp)
    
    def sqrt(self):
        return self._apply_function(Value.sqrt)
    
    def abs(self):
        return self._apply_function(Value.abs)
    
    def backward(self):
        for item in self:
            item.backward()
    
    def __repr__(self):
        return f'Array({[item for item in self]})'
    
    @staticmethod
    def arg_max(array):
        if not array:
            raise ValueError("Array is empty")
        max_index = 0
        max_value = array[0].data
        for i in range(1, len(array)):
            if array[i].data > max_value:
                max_value = array[i].data
                max_index = i
        return max_index
    
    @staticmethod
    def arg_min(array):
        if not array:
            raise ValueError("Array is empty")
        min_index = 0
        min_value = array[0].data
        for i in range(1, len(array)):
            if array[i].data < min_value:
                min_value = array[i].data
                min_index = i
        return min_index
    
    @staticmethod
    def zeros_like(array):
        return Array([0] * len(array))
    
    @staticmethod
    def zeros(l: int):
        return Array([0] * l)

    def max(self):
        return self[Array.arg_max(self)]
    
    def min(self):
        return self[Array.arg_min(self)]

    def avg(self):
        return sum(self)/len(self)