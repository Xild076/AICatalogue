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
    
    def _shape(self, tensor):
        if not isinstance(tensor, list):
            return ()
        return (len(tensor),) + self._shape(tensor[0])
    
    def _map_data(self, func, other=None):
        pass
    
    def _check_compatibility(self, a, b):
        if len(a.shape) == 0  and len(b.shape) == 0:
            return 0
        elif len(a.shape) == 0 and len(b.shape) == 1 or len(a.shape) == 1 and len(b.shape) == 0:
            return 1
        
        
    
    def __add__(self, other):
        other = other if isinstance(other, value) else value(other)
        out = value()