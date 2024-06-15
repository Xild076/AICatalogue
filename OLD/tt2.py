import math

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = self._to_nested_list(data)
        self.grad = self._zero_grad(self.data)
        self._backward = lambda: None
        self._prev = _children
        self._op = _op
        self.shape = self._shape(self.data)

    def _to_nested_list(self, data):
        if isinstance(data, (int, float)):
            return data
        return [self._to_nested_list(x) for x in data]

    def _zero_grad(self, data):
        if isinstance(data, (int, float)):
            return 0
        return [self._zero_grad(x) for x in data]

    def _map_data(self, func, other=None):
        if isinstance(self.data, (int, float)):
            if other is None:
                return func(self.data)
            return func(self.data, other)
        if other is None:
            return [Value._map_data(self._copy_child(x), func) for x in self.data]
        return [Value._map_data(self._copy_child(x), func, y) for x, y in zip(self.data, other)]

    def _copy_child(self, child):
        if isinstance(child, Value):
            return child
        return Value(child)
    
    def _map_grad(self, func, *args):
        if isinstance(self.grad, (int, float)):
            return func(self.grad, *args)
        return [Value._map_grad(x, func, *args) for x in self.grad]

    def _map_zip(self, func, a, b):
        # If a is scalar and b is list, broadcast a to match the structure of b
        if isinstance(a, (int, float)) and isinstance(b, list):
            a = [a] * len(b)
        # If b is scalar and a is list, broadcast b to match the structure of a
        if isinstance(b, (int, float)) and isinstance(a, list):
            b = [b] * len(a)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return func(a, b)
        if isinstance(a, (list)) and isinstance(b, (list)):
            return [self._map_zip(func, x, y) for x, y in zip(a, b)]
        raise ValueError("Shape mismatch for element-wise operation.")

    def _add_helper(self, a, b):
        if isinstance(a, (int, float)):
            return a + b
        return [self._add_helper(x, y) for x, y in zip(a, b)]

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self._map_zip(lambda x, y: x + y, self.data, other.data), (self, other), '+')

        def _backward():
            self.grad = self._map_zip(lambda x, y: x + y, self.grad, out.grad)
            other.grad = other._map_zip(lambda x, y: x + y, other.grad, out.grad)
        out._backward = _backward
        
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self._map_zip(lambda x, y: x * y, self.data, other.data), (self, other), '*')
        
        def _backward():
            self.grad = self._map_zip(lambda x, y: x + y * out.grad, self.grad, other.data)
            other.grad = other._map_zip(lambda x, y: x + y * out.grad, other.grad, self.data)
        out._backward = _backward
        
        return out

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self._map_data(lambda x: x ** other.data), (self,), f'**{other}')
        
        def _backward():
            self.grad = self._map_zip(lambda x, y: x + y * out.grad, self.grad, self._map_data(lambda x: other.data * x ** (other.data - 1)))
        out._backward = _backward

        return out
    
    def _transpose(self, tensor):
        if isinstance(tensor[0], (int, float)):
            return tensor
        return [list(row) for row in zip(*tensor)]

    def __matmul__(self, other):
        if isinstance(self.data[0], (int, float)) and isinstance(other.data[0], (int, float)):
            return sum(self.data[i] * other.data[i] for i in range(len(self.data)))
        result = [[sum(a * b for a, b in zip(row, col)) for col in zip(*other.data)] for row in self.data]
        out = Value(result, (self, other), '@')

        def _backward():
            self.grad = self._map_zip(lambda x, y: x + y, self.grad, out.grad)  # Simplified
            other.grad = self._map_zip(lambda x, y: x + y, other.grad, out.grad)  # Simplified
        out._backward = _backward
        
        return out

    def relu(self):
        out_data = self._map_data(lambda x: max(0, x))
        out = Value(out_data, (self,), 'ReLU')
        
        def _backward():
            self.grad = self._map_zip(lambda x, y: x + (y > 0) * out.grad, self.grad, out.data)
        out._backward = _backward
        
        return out

    def tanh(self):
        t = self._map_data(lambda x: math.tanh(x))
        out = Value(t, (self,), 'Tanh')
        
        def _backward():
            self.grad = self._map_zip(lambda x, y: x + (1 - y ** 2) * out.grad, self.grad, out.data)
        out._backward = _backward
        
        return out
    
    def sigmoid(self):
        s = self._map_data(lambda x: 1 / (1 + math.exp(-x)))
        out = Value(s, (self,), 'Sigmoid')
        
        def _backward():
            self.grad = self._map_zip(lambda x, y: x + y * (1 - y) * out.grad, self.grad, out.data)
        out._backward = _backward
        
        return out
    
    def log(self):
        out_data = self._map_data(lambda x: math.log(x + 1e-8))
        out = Value(out_data, (self,), 'Log')
        
        def _backward():
            self.grad = self._map_zip(lambda x, y: x + 1 / (y + 1e-8) * out.grad, self.grad, self.data)
        out._backward = _backward
        
        return out
    
    def sqrt(self):
        return self ** 0.5
    
    def exp(self):
        out = self._map_data(lambda x: math.exp(x))
        out = Value(out, (self,), 'Exp')

        def _backward():
            self.grad = self._map_zip(lambda x, y: x + y * out.grad, self.grad, out.data)
        out._backward = _backward

        return out
    
    def abs(self):
        out_data = self._map_data(lambda x: abs(x))
        out = Value(out_data, (self,), 'Abs')
        
        def _backward():
            self.grad = self._map_zip(lambda x, y: x + (1 if y > 0 else -1) * out.grad, self.grad, self.data)
        out._backward = _backward
        
        return out

    def __abs__(self):
        return self.abs()
    
    def __repr__(self):
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
        self.grad = self._zero_grad(self.data)
        self.grad = self._map_data(lambda _: 1)
        
        for v in reversed(topo):
            v._backward()

    def _shape(self, tensor):
        if not isinstance(tensor, list):
            return ()
        return (len(tensor),) + self._shape(tensor[0])
    


t = Value([[1, 3, 2], [3, 2, 1], [1, 4, 2]])
b = Value([[3, 2, 1], [1, 4, 2], [3, 1, 2]])
print(t.shape)
c = Value(1)
print(c.shape())
print(len(c.shape()))
"""
c = t * b

print(c)
c.backward()
print(t)"""