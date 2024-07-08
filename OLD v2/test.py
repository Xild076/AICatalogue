from pg import pg
from engine import value
from env import jump_env
from optim import SGD
from nn import MLP, Neuron, Layer
from dt import Categorical


"""je = jump_env()
print('JE AS', je.action_space)
print('JE OS', je.observation_space)"""

"""m = pg([2, 8, 5], ['relu', ''], SGD, 0.001)
m.train(je, 10000)"""

l = MLP([2, 3, 2], ['', ''])
inp = [value([4, 2]), value([3, 4]), value([-1, 1]), value([5, -2])]
o = l(inp)
print(o)
s = o.softmax()
print(s)
c = Categorical(probs=s)
print(c.sample())
lp = c.log_prob(c.sample())
print(lp)
su = lp.to_scalar()
print(su)
su.backward()
print('w', l.layers[0].neurons[0].w)
print('o', o)
print('s', s)