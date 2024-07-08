import math
from engine import Value, Array


"""def softmax(array):
    if isinstance(array, list):
        array = Array(array)
    max_val = max(array, key=lambda x: x.data).data
    exp_vals = [math.exp(v.data - max_val) for v in array]
    sum_exp_vals = sum(exp_vals)
    out = Array(Value(v / sum_exp_vals) for v in exp_vals)
    
    def _backward():
        for i, v in enumerate(out):
            s = v.data
            array[i].grad += s * (1 - s) * v.grad
            for j, u in enumerate(out):
                if i != j:
                    array[j].grad -= s * u.data * v.grad

    for v in out:
        v._backward = _backward
    
    return out
"""

def softmax(array):
    if isinstance(array, list):
        array = Array(array)
    
    exp_values = array.exp()
    sum_exp_values = sum(exp_values)
    softmax_values = exp_values / sum_exp_values
    
    return softmax_values


