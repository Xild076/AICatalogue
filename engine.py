import numpy as np

class value:
    _no_grad = False

    def __init__(self, data, _children=(), _op='', _ng=False):
        self.data = np.array(data, dtype='float64') if not isinstance(data, type(np.array([]))) else data
        self.grad = np.zeros_like(self.data)
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None
        self.shape = self.data.shape
        self._ng = _ng

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def __add__(self, other):
        other = other if isinstance(other, value) else value(other)
        out = value(self.data + other.data, (self, other), '+', value._no_grad)

        def _backward():
            if out._ng:
                return
            self_grad = out.grad
            other_grad = out.grad
            if self.shape != other.shape:
                self_grad = np.sum(self_grad, axis=tuple(range(self_grad.ndim - self.data.ndim)))
                other_grad = np.sum(other_grad, axis=tuple(range(other_grad.ndim - other.data.ndim)))
                self_grad = np.reshape(self_grad, self.shape)
                other_grad = np.reshape(other_grad, other.shape)
            self.grad += self_grad
            other.grad += other_grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, value) else value(other)
        out = value(self.data * other.data, (self, other), '*', value._no_grad)

        def _backward():
            if out._ng:
                return
            self_grad = other.data * out.grad
            other_grad = self.data * out.grad
            if self.shape != other.shape:
                self_grad = np.sum(self_grad, axis=tuple(range(self_grad.ndim - self.data.ndim)))
                other_grad = np.sum(other_grad, axis=tuple(range(other_grad.ndim - other.data.ndim)))
                self_grad = np.reshape(self_grad, self.shape)
                other_grad = np.reshape(other_grad, other.shape)
            self.grad += self_grad
            other.grad += other_grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Power must be a scalar"
        out = value(self.data ** other, (self,), f'**{other}', value._no_grad)

        def _backward():
            if out._ng:
                return
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return self - other

    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __rtruediv__(self, other):
        return self / other

    def __getitem__(self, key):
        out = value(self.data[key], (self,), 'getitem', value._no_grad)

        def _backward():
            if out._ng:
                return
            grad = np.zeros_like(self.data)
            np.add.at(grad, key, out.grad)
            self.grad += grad
        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, value) else value(other)
        out = value(np.matmul(self.data, other.data), (self, other), '@', value._no_grad)

        def _backward():
            if out._ng:
                return
            self.grad += np.matmul(out.grad, other.data.T)
            other.grad += np.matmul(self.data.T, out.grad)
        out._backward = _backward
        return out

    def transpose(self, axes=None):
        out = value(np.transpose(self.data, axes), (self,), 'transpose', value._no_grad)

        def _backward():
            if out._ng:
                return
            self.grad += np.transpose(out.grad, axes)
        out._backward = _backward
        return out

    def clip(self, min=None, max=None):
        min = min.data if isinstance(min, value) else min
        max = max.data if isinstance(max, value) else max
        out = value(np.clip(self.data, min, max), (self,), 'clip', value._no_grad)

        def _backward():
            if out._ng:
                return
            mask = (self.data >= min) & (self.data <= max)
            self.grad += mask * out.grad
        out._backward = _backward
        return out

    def abs(self):
        out = value(np.abs(self.data), (self,), 'abs', value._no_grad)

        def _backward():
            if out._ng:
                return
            self.grad += (self.data / (np.abs(self.data) + 1e-12)) * out.grad
        out._backward = _backward
        return out

    def sqrt(self):
        out = value(np.sqrt(self.data), (self,), 'sqrt', value._no_grad)

        def _backward():
            if out._ng:
                return
            self.grad += (0.5 / out.data) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = value(np.maximum(0, self.data), (self,), 'relu', value._no_grad)

        def _backward():
            if out._ng:
                return
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = value(np.log(self.data + 1e-8), (self,), 'log', value._no_grad)

        def _backward():
            if out._ng:
                return
            self.grad += (1 / (self.data + 1e-8)) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = value(np.exp(self.data), (self,), 'exp', value._no_grad)

        def _backward():
            if out._ng:
                return
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def sinh(self):
        out = value(np.sinh(self.data), (self,), 'sinh', value._no_grad)

        def _backward():
            if out._ng:
                return
            self.grad += (np.cosh(self.data)) * out.grad
        out._backward = _backward
        return out

    def cosh(self):
        out = value(np.cosh(self.data), (self,), 'cosh', value._no_grad)

        def _backward():
            if out._ng:
                return
            self.grad += (np.sinh(self.data)) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        out = value(np.tanh(self.data), (self,), 'tanh', value._no_grad)

        def _backward():
            if out._ng:
                return
            self.grad += (1 - np.tanh(self.data) ** 2) * out.grad
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = value(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), 'sum', value._no_grad)

        def _backward():
            if out._ng:
                return
            if axis is None:
                expanded_grad = np.ones_like(self.data) * out.grad
            else:
                expanded_grad = np.expand_dims(out.grad, axis)
                expanded_grad = np.broadcast_to(expanded_grad, self.shape)
            self.grad += expanded_grad
        out._backward = _backward
        return out
    
    def mean(self, axis=None, keepdims=False):
        out = value(np.mean(self.data, axis=axis, keepdims=keepdims), (self,), 'mean', value._no_grad)

        def _backward():
            if out._ng:
                return
            if axis is None:
                num_elements = self.data.size
                expanded_grad = np.ones_like(self.data) * out.grad / num_elements
            else:
                num_elements = self.data.shape[axis]
                expanded_grad = np.expand_dims(out.grad, axis)
                expanded_grad = np.broadcast_to(expanded_grad, self.shape) / num_elements
            self.grad += expanded_grad
        
        out._backward = _backward
        return out

    def std(self, axis=None, keepdims=False):
        mean = self.mean()
        variance = ((self - mean) ** 2).mean(axis=axis, keepdims=keepdims)
        out = value(np.sqrt(variance.data), (self,), 'std', value._no_grad)

        def _backward():
            if out._ng:
                return
            n = len(self)
            var_grad = (self.data - mean.data) / (n * out.data) * out.grad
            self.grad += var_grad
        out._backward = _backward
        return out
    
    def softmax(self):
        exps = np.exp(self.data - np.max(self.data, axis=-1, keepdims=True))
        sum_exps = np.sum(exps, axis=-1, keepdims=True)
        out = value(exps / sum_exps, (self,), 'softmax', value._no_grad)

        def _backward():
            if out._ng:
                return
            s = out.data
            if s.ndim == 1:
                jacobian_matrix = np.diagflat(s) - np.outer(s, s)
                self.grad += np.dot(jacobian_matrix, out.grad)
            else:
                jacobian_matrix = np.empty((s.shape[0], s.shape[1], s.shape[1]))
                for i in range(s.shape[0]):
                    jacobian_matrix[i] = np.diagflat(s[i]) - np.outer(s[i], s[i])
                self.grad += np.einsum('ijk,ik->ij', jacobian_matrix, out.grad)

        out._backward = _backward
        return out

    def argmax(self, axis=None):
        return np.argmax(self.data, axis=axis)

    def argmin(self, axis=None):
        return np.argmin(self.data, axis=axis)

    def max(self, axis=None, keepdims=False):
        out = value(np.max(self.data, axis=axis, keepdims=keepdims), (self,), 'max', value._no_grad)

        def _backward():
            if out._ng:
                return
            mask = self.data == out.data
            self.grad += mask * out.grad
        out._backward = _backward
        return out

    def min(self, axis=None, keepdims=False):
        out = value(np.min(self.data, axis=axis, keepdims=keepdims), (self,), 'min', value._no_grad)

        def _backward():
            if out._ng:
                return
            mask = self.data == out.data
            self.grad += mask * out.grad
        out._backward = _backward
        return out
    
    def amax(self, axis=None, keepdims=False):
        out = value(np.amax(self.data, axis=axis, keepdims=keepdims), (self,), 'amax', value._no_grad)

        def _backward():
            if out._ng:
                return
            grad = np.zeros_like(self.data)
            mask = (self.data == out.data)
            grad[mask] = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            self.grad += grad
        out._backward = _backward
        return out

    def amin(self, axis=None, keepdims=False):
        out = value(np.amin(self.data, axis=axis, keepdims=keepdims), (self,), 'amin', value._no_grad)

        def _backward():
            if out._ng:
                return
            grad = np.zeros_like(self.data)
            mask = (self.data == out.data)
            grad[mask] = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            self.grad += grad
        out._backward = _backward
        return out

    def maximum(self, other):
        other = other if isinstance(other, value) else value(other)
        out = value(np.maximum(self.data, other.data), (self, other), 'maximum', value._no_grad)

        def _backward():
            if out._ng:
                return
            self_mask = (self.data >= other.data)
            other_mask = (self.data < other.data)
            self.grad += self_mask * out.grad
            other.grad += other_mask * out.grad

        out._backward = _backward
        return out

    def minimum(self, other):
        other = other if isinstance(other, value) else value(other)
        out = value(np.minimum(self.data, other.data), (self, other), 'minimum', value._no_grad)

        def _backward():
            if out._ng:
                return
            self_mask = (self.data <= other.data)
            other_mask = (self.data > other.data)
            self.grad += self_mask * out.grad
            other.grad += other_mask * out.grad

        out._backward = _backward
        return out

    def flatten(self):
        out = value(self.data.flatten(), (self,), 'flatten', value._no_grad)

        def _backward():
            if out._ng:
                return
            self.grad += out.grad.reshape(self.shape)
        out._backward = _backward
        return out

    @staticmethod
    def join(values):
        if not values:
            return value(0)
        combined_data = np.concatenate([v.data.flatten() for v in values])
        combined_value = value(combined_data, values, 'combine', value._no_grad)

        def _backward():
            if combined_value._ng:
                return
            start = 0
            for v in values:
                size = v.data.size
                v.grad += combined_value.grad[start:start + size].reshape(v.data.shape)
                start += size

        combined_value._backward = _backward
        return combined_value
    
    @staticmethod
    def stack(values):
        if not values:
            return value(0)
        stacked_data = np.stack([v.data for v in values])
        stacked_value = value(stacked_data, values, 'stack', value._no_grad)

        def _backward():
            if stacked_value._ng:
                return
            for i, v in enumerate(values):
                v.grad += stacked_value.grad[i]

        stacked_value._backward = _backward
        return stacked_value
    
    @staticmethod
    def transform(values):
        if not values:
            return value(0)
        joined_data = np.array([v.data for v in values])
        joined_value = value(joined_data, values, 'join', value._no_grad)

        def _backward():
            if joined_value._ng:
                return
            for i, v in enumerate(values):
                v.grad += joined_value.grad[i]
        joined_value._backward = _backward
        return joined_value
    
    def unsqueeze(self, axis):
        out = value(np.expand_dims(self.data, axis), (self,), 'unsqueeze', value._no_grad)

        def _backward():
            if out._ng:
                return
            self.grad += np.squeeze(out.grad, axis)
        out._backward = _backward
        return out
    
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
        self.grad = np.ones_like(self.data)

        for v in reversed(topo):
            v._backward()

    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        try:
            return f"value(data={self.data.tolist()}, grad={self.grad.tolist()})"
        except:
            return f"value(data={self.data}, grad={self.grad})"

    def __hash__(self):
        return hash(self.data.tobytes())

    def __eq__(self, other):
        if isinstance(other, value):
            return np.array_equal(self.data, other.data)
        return False

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        other = other if isinstance(other, value) else value(other)
        return np.all(self.data < other.data)

    def __le__(self, other):
        other = other if isinstance(other, value) else value(other)
        return np.all(self.data <= other.data)

    def __gt__(self, other):
        other = other if isinstance(other, value) else value(other)
        return np.all(self.data > other.data)

    def __ge__(self, other):
        other = other if isinstance(other, value) else value(other)
        return np.all(self.data >= other.data)

    @staticmethod
    def no_grad():
        class NoGradContextManager:
            def __enter__(self):
                value._no_grad = True

            def __exit__(self, exc_type, exc_value, traceback):
                value._no_grad = False

        return NoGradContextManager()
