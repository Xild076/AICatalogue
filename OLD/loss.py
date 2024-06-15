from engine import Array, Value
from functional import softmax
from cd import Categorical


def MSE(pred: Array, real: Array):
    return ((real - pred) ** 2).avg()

def MAE(pred: Array, real: Array):
    return (real - pred).abs().avg()

def BinaryCrossEntropy(pred: Array, real: Array):
    return -(real * pred.log() + (1 - real) * (1 - pred).log()).avg()

def CrossEntropy(pred: Array, real: Array):
    return -(real * pred.log()).avg() 

