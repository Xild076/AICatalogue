import torch
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.nn as nn

a = torch.tensor([1.0, 3.0, 2.0, 4.0], requires_grad=True)

b = F.softmax(a, dim=0)
b.retain_grad()

c = Categorical(logits=a)

d = c.log_prob(torch.tensor(3))
d.retain_grad()

e = d * 3
e.retain_grad()

f = -e
f.retain_grad()

g = f - torch.tensor(3)
g.retain_grad()

h = torch.relu(g)
h.retain_grad()

i = h + torch.tensor(1)
i.retain_grad()

g.backward()

print("Gradient of a:", a.grad)
print("Gradient of b:", b.grad)
print("Gradient of d:", d.grad)
print("Gradient of e:", e.grad)
print("Gradient of f:", f.grad)
print("Gradient of g:", g.grad)
print("Gradient of h:", h.grad)
print("Gradient of i:", i.grad)


from engine import Value, Array
from functional import softmax
from cd import Categorical

a = Array(1, 3, 2, 4)
b = softmax(a)
c = Categorical(a)
d = c.log_prob(3)
e = d * 3
f = -e
g = f - 3
h = g.relu()
i = h + 1

g.backward()

print("Gradient of a:", a)
print("Gradient of b:", b)
print("Gradient of d:", d)
print("Gradient of e:", e)
print("Gradient of f:", f)
print("Gradient of g:", g)
print("Gradient of h:", h)
print("Gradient of i:", i)

print('\n\n\n\n')

import torch
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.nn as nn


x = torch.tensor(3.0, requires_grad=True)
x.retain_grad()

t = torch.tensor([4.0, -3.0, 2.0], requires_grad=True)
t.retain_grad()

t1 = x * t
t1.retain_grad()

t2 = F.relu(t1)
t2.retain_grad()

st = F.softmax(t2, dim=0)
st.retain_grad()

cat = Categorical(st)

lp = cat.log_prob(cat.sample())
lp.retain_grad()

r = torch.tensor(0.99)

l = -lp * r
l.backward()

print("L:", l, "| Grad:", l.grad)
print("lp:", lp, "| Grad:", lp.grad)
print("st:", st, "| Grad:", st.grad)
print("t2:", t2, "| Grad:", t2.grad)
print("t1:", t1, "| Grad:", t1.grad)
print("t:", t, "| Grad:", t.grad)
print("x:", x, "| Grad:", x.grad)
print("")


from engine import Value, Array
from functional import softmax
from cd import Categorical


x = Value(3)
t = Array(4, -3, 2)
t1 = t * x
t2 = t1.relu()
st = softmax(t2)
cat = Categorical(st, probs=True)
lp = cat.log_prob(cat.sample())
r = 0.99
l = -lp * r
l.backward()

print("L:", l)
print("lp:", lp)
print("st:", st)
print("t2:", t2)
print("t1:", t1)
print("t:", t)
print("x:", x)