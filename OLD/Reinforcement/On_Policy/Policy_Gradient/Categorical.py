import PG
import random
import numpy as np
import time


class Categorical(PG.Model):
    def __init__(self, config: dict, layer: list, act: list):
        super().__init__(config, layer, act)
    
    def sample_action(self, x):
        e_x = np.exp(x - np.max(x))
        e_x = e_x / e_x.sum(axis=0)
        if random.random() < 0.05:
            act = random.randint(0, len(x) - 1)
        else:
            act = np.argmax(e_x)
        aoh = np.zeros_like(x)
        aoh[act] = 1
        a_g = aoh - e_x
        
        return act, a_g


class Gaussian(PG.Model):
    def __init__(self, config: dict, layer: list, act: list):
        super().__init__(config, layer, act)
    
    def sample_action(self, x):
        noise = np.random.normal(loc=0, scale=0.1, size=len(x))
        return x + noise, x