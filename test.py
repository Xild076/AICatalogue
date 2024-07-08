from engine import value
from nn import MLP


m = MLP([4, 2, 1], ['relu', ''])

v = value([[2, 4, 5, 1], [2, 4, 1, 5]])
o = m(v)

print(o)