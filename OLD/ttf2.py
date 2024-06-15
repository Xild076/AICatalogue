import math


class value:
    def __init__(self, data, _children=(), _op=''):
        self.data = self._nested(data)
        self.grad = self._zero_grad(self.data)
        self._prev = _children
        self._backward = lambda: None
        self._op = _op
        self.shape = self._shape(self.data)
    
    def _nested(self, data):
        if isinstance(data, (int, float)):
            return data
        return [self._nested(x) for x in data]
    
    def _zero_grad(self, data):
        if isinstance(data, (int, float)):
            return 0
        return [self._zero_grad(x) for x in data]
    
    def _shape(self, t):
        if not isinstance(t, list):
            return ()
        return (len(t),) + self._shape(t[0])
    
    def _check_compatibility(self, ad, bd):
        a = list(self._shape(ad))
        b = list(self._shape(bd))
        if len(a) == 0 and len(b) == 0:
            return 0
        if len(a) == 0:
            return 1
        if len(b) == 0:
            return 2
        b_dim_down = b.copy()
        b_dim_down.pop(-1)
        if a == b_dim_down:
            return 1
        a_dim_down = b.copy()
        a_dim_down.pop(-1)
        if b == a_dim_down:
            return 2
        if len(a) == len(b):
            return 3
        raise ValueError("Shape mismatch for element-wise operation.")
    
    def _map_zip(self, func, a, b):
        cc = self._check_compatibility(a, b)
        
        if cc == 0:
            return func(a, b)
        if cc == 1:
            a = [a] * len(b)
        if cc == 2:
            b = [b] * len(a)
        return [self._map_zip(func, x, y) for x, y in zip(a, b)]
    
    def _map_data(self, func, other=None):
        if isinstance(self.data, (int, float)):
            if other is None:
                return func(self.data)
            return func(self.data, other)
        if other is None:
            return [value._map_data(self._copy_child(x), func) for x in self.data]
        return [value._map_data(self._copy_child(x), func, y) for x, y in zip(self.data, other)]

    def _copy_child(self, child):
        if isinstance(child, value):
            return child
        return value(child)
    
    def __add__(self, other):
        other = other if isinstance(other, value) else value(other)
        out = value(self._map_zip(lambda x, y: x + y, self.data, other.data), (self, other), '+')
        
        def _backward():
            self.grad = self._map_zip(lambda x, y: x + y, self.grad, out.grad)
            other.grad = self._map_zip(lambda x, y: x + y, other.grad, out.grad)
        out._backward = _backward
        
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, value) else value(other)
        out = value(self._map_zip(lambda x, y: x * y, self.data, other.data), (self, other), '*')
        
        def _backward():
            self.grad = self._map_zip(lambda x, y: x + y, self.grad, self._map_zip(lambda x, y: x * y, other.data, out.grad))
            other.grad = self._map_zip(lambda x, y: x + y, other.grad, self._map_zip(lambda x, y: x * y, self.data, out.grad))
        out._backward = _backward
        
        return out

    def __pow__(self, other):
        out = value(self._map_data(lambda x: x ** other), (self,), f'**{other}')
        
        def _backward():
            self.grad = self._map_zip(lambda x, y: x + y, self.grad, self._map_zip(lambda x, y: x * y, out.grad, self._map_data(lambda x: other * x ** (other - 1))))
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
        out = value(self._map_data(lambda x: max(0, x)), (self,), 'relu')
        
        def _backward():
            self.grad = self._map_zip(lambda x, y: x + y, self.grad, self._map_zip(lambda x, y: (x > 0) * y, out.data, out.grad))
        out._backward = _backward
        
        return out
    
    def log(self):
        out = value(self._map_data(lambda x: math.log(x + 1e-8)), (self,), 'log')
        
        def _backward():
            self.grad = self._map_zip(lambda x, y: x + y, self.grad, self._map_zip(lambda x, y: 1 / (x + 1e-8) * y, self.data, out.grad))
        out._backward = _backward
        
        return out
    
    def exp(self):
        out = value(self._map_data(lambda x: math.exp(x)), (self,), 'exp')
        
        def _backward():
            self.grad = self._map_zip(lambda x, y: x + y, self.grad, self._map_zip(lambda x, y: x * y, out.data, out.grad))
        out._backward = _backward
        
        return out
    
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


a = value([[-1, 3, 2], [3, 1, 2]])
b = value([[1, 3, 2]])

print(a + b)

c = a.exp()
print(a / 2)
c.backward()
print(a)