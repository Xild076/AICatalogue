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

    def zero_grad(self):
        self.grad = self._zero_grad(self.data)
    
    def _shape(self, t):
        if not isinstance(t, list):
            return ()
        return (len(t),) + self._shape(t[0])
    
    def _check_compatibility(self, ad, bd):
        a = self._shape(ad)
        b = self._shape(bd)
                
        if not a and not b:
            return 0

        if not a:
            return 1
        
        if not b:
            return 2
        
        if a == b[:-1]:
            return 1
        
        if b == a[:-1]:
            return 2
        
        if a == b[1:]:
            return 1
        
        if b == a[1:]:
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
    
    def _collapse_grad_singular(self, grad, out_grad):
        grad_shape = self._shape(grad)
        out_grad_shape = self._shape(out_grad)
        
        if grad_shape == out_grad_shape[1:] and grad_shape:
            out_grad = self._transpose(out_grad)
            out_grad_shape = self._shape(grad_shape)
        
        while not out_grad_shape == grad_shape:
            out_grad = self._in_lay(out_grad)
            out_grad_shape = self._shape(out_grad)
        
        return out_grad
    
    def _collapse_grad_dual(self, self_grad, other_grad, out_grad):
        return self._collapse_grad_singular(self_grad, out_grad), self._collapse_grad_singular(other_grad, out_grad)

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
            
            self_grad, other_grad = self._collapse_grad_dual(self.grad, other.grad, out.grad)
            
            self.grad = self._map_zip(lambda x, y: x + y, self.grad, self_grad)
            other.grad = self._map_zip(lambda x, y: x + y, other.grad, other_grad)
        out._backward = _backward
        
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, value) else value(other)
        out = value(self._map_zip(lambda x, y: x * y, self.data, other.data), (self, other), '*')

        def _backward():
            
            self_grad = self._map_zip(lambda x, y: x * y, other.data, out.grad)
            other_grad = self._map_zip(lambda x, y: x * y, self.data, out.grad)
            
            self_grad = self._collapse_grad_singular(self.grad, self_grad)
            other_grad = self._collapse_grad_singular(other.grad, other_grad)
                        
            self.grad = self._map_zip(lambda x, y: x + y, self.grad, self_grad)
            other.grad = self._map_zip(lambda x, y: x + y, other.grad, other_grad)
        
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
    
    def _in_lay(self, datas):
        if isinstance(datas[0], (int, float)):
            return sum(datas)
        return [self._in_lay(data) for data in datas]
    
    def sum(self):
        out = value(self._in_lay(self.data), (self,))
        
        def _backward():
            self_grad = self._map_data(lambda _: 1)
            self.grad = self._map_zip(lambda x, y: x + y, self.grad, self_grad)
        out._backward = _backward
        
        return out
    
    def to_scalar(self):
        def _collapse(data):
            if isinstance(data, (int, float)):
                return data
            return sum(_collapse(item) for item in data)
        
        out = value(_collapse(self.data), (self,), 'collapse')
        
        def _backward():
            self_grad = self._map_data(lambda _: out.grad)
            self.grad = self._map_zip(lambda x, y: x + y, self.grad, self_grad)
        out._backward = _backward
        
        return out

    def _flatten(self, data):
        if isinstance(data, (int, float)):
            return [data]
        return [item for sublist in data for item in self._flatten(sublist)]
    
    def __repr__(self):
        rnd = self.round_nested(self.data, 3)
        rng = self.round_nested(self.grad, 3)
        return f"Value(data={rnd}, grad={rng})"
    
    def round_nested(self, data, decimals=0):
        if isinstance(data, list):
            return [self.round_nested(item, decimals) for item in data]
        return round(data, decimals)
    
    def __len__(self):
        if isinstance(self.data, (list)):
            return len(self.data)
        return -1
    
    @staticmethod
    def combine(values):
        if not values:
            return value(0)

        combined_data = [v.data for v in values]      
        combined_grads = [v.grad for v in values]

        combined_value = value(combined_data, set(values))
        combined_value.grad = combined_grads
        
        def _backward():
            for v in values:
                v.grad = combined_value.grad[values.index(v)]
                v._backward()
        combined_value._backward = _backward

        return combined_value
    
    @staticmethod
    def split(combined_value):
        if not isinstance(combined_value, value) or not isinstance(combined_value.data, list):
            raise ValueError("Input must be a combined value object with a list of data.")

        data_list = combined_value.data
        grad_list = combined_value.grad

        split_values = []
        for data, grad in zip(data_list, grad_list):
            new_value = value(data)
            new_value.grad = grad
            
            def _backward(new_value=new_value, combined_value=combined_value):
                combined_value.grad[combined_value.data.index(new_value.data)] = new_value.grad
                combined_value._backward()
            
            new_value._backward = _backward
            split_values.append(new_value)

        return split_values
    
    def step(self, grad):
        self.data = self._map_zip(lambda x, y: x - y, self.data, grad)
    
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

    def softmax(self):
        exp_values = self.exp()
        sum_exp_values = exp_values.sum()
        if len(self.shape) > 1:
            exp_values = exp_values.transpose()
            softmax_values = exp_values / sum_exp_values
            softmax_values = softmax_values.transpose()
        else:
            softmax_values = exp_values / sum_exp_values
        return softmax_values
    
    def argmax(self):
        flattened_data = self._flatten(self.data)
        max_index = flattened_data.index(max(flattened_data))
        return max_index
    
    def argmin(self):
        flattened_data = self._flatten(self.data)
        min_index = flattened_data.index(min(flattened_data))
        return min_index
    
    def max(self):
        loc = self.argmax()
        out = value(self.data[loc], (self,), 'max')

        def _backward():
            self.grad[loc] += out.grad
        out._backward = _backward
        return out

    def min(self):
        loc = self.argmin()
        out = value(self.data[loc], (self,), 'min')

        def _backward():
            self.grad[loc] += out.grad
        out._backward = _backward
        return out

    def __getitem__(self, index):
        out = value(self.data[index], (self,), 'getitem')

        def _backward():
            self.grad[index] += out.grad
        out._backward = _backward
        return out
    
    def _transpose(self, tensor):
        if isinstance(tensor[0], (int, float)):
            return tensor
        return [list(row) for row in zip(*tensor)]

    def transpose(self):
        transposed_data = self._transpose(self.data)
        transposed_grad = self._transpose(self.grad)
                
        out = value(transposed_data, (self,), 'transpose')
        out.grad = transposed_grad
                
        return out
    
    def __matmul__(self, other):
        other = other if isinstance(other, value) else value(other)
        
        if len(self.shape) == 2 and len(other.shape) == 2:
            if self.shape[1] != other.shape[0]:
                raise ValueError(f"Incompatible shapes for matrix multiplication: {self.shape} and {other.shape}")
            result = [[sum(a * b for a, b in zip(row, col)) for col in zip(*other.data)] for row in self.data]
        elif len(self.shape) == 2 and len(other.shape) == 1:
            if self.shape[1] != other.shape[0]:
                raise ValueError(f"Incompatible shapes for matrix-vector multiplication: {self.shape} and {other.shape}")
            result = [sum(a * b for a, b in zip(row, other.data)) for row in self.data]
        elif len(self.shape) == 1 and len(other.shape) == 2:
            if self.shape[0] != other.shape[0]:
                raise ValueError(f"Incompatible shapes for vector-matrix multiplication: {self.shape} and {other.shape}")
            result = [sum(a * b for a, b in zip(self.data, col)) for col in zip(*other.data)]
        elif len(self.shape) == 1 and len(other.shape) == 1:
            return self * other
        else:
            raise ValueError("Incompatible shapes for matrix multiplication: {self.shape} and {other.shape}")

        out = value(result, (self, other), '@')
        
        def _backward():
            if len(self.shape) == 2 and len(other.shape) == 2:
                grad_self = [[sum(out.grad[i][k] * other.data[j][k] for k in range(other.shape[1])) for j in range(self.shape[1])] for i in range(self.shape[0])]
                grad_other = [[sum(self.data[k][i] * out.grad[k][j] for k in range(self.shape[0])) for j in range(other.shape[1])] for i in range(other.shape[0])]
            elif len(self.shape) == 2 and len(other.shape) == 1:
                grad_self = [[out.grad[i] * other.data[j] for j in range(self.shape[1])] for i in range(self.shape[0])]
                grad_other = [sum(self.data[i][j] * out.grad[i] for i in range(self.shape[0])) for j in range(self.shape[1])]
            elif len(self.shape) == 1 and len(other.shape) == 2:
                grad_self = [sum(out.grad[j] * other.data[i][j] for j in range(other.shape[1])) for i in range(self.shape[0])]
                grad_other = [[self.data[i] * out.grad[j] for j in range(other.shape[1])] for i in range(other.shape[0])]
            else:
                raise ValueError("Incompatible shapes for matrix multiplication: {self.shape} and {other.shape}")
            
            self.grad = self._map_zip(lambda x, y: x + y, self.grad, grad_self)
            other.grad = self._map_zip(lambda x, y: x + y, other.grad, grad_other)
        
        out._backward = _backward
                
        return out

