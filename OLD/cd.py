from engine import Value, Array
from functional import softmax
import types
import copy
import math
import random


class Categorical(Array):
    def __init__(self, *args, probs=False):
        super().__init__(*args)
        
        if not probs:
            self.logits = Array(item for item in self)
            self._logit_prob()
        else:
            self._prob_logit()
    
    def _logit_prob(self):
        probs = softmax(self)
        
        for i in range(len(self)):
            self[i] = probs[i]
    
    def _prob_logit(self):
        self.logits = Array(item.log() for item in self)
    
    def __repr__(self):
        return f'Array(logits={[item for item in self.logits]}, probs={[item for item in self]})' 

    def log_prob(self, value):
        max_logit = self.logits.max()
        logsumexp = max_logit + sum((self.logits - max_logit).exp()).log()
        return self.logits[value] - logsumexp
    
    def sample(self, rand=0.0):
        if random.random() <= rand:
            return random.randint(0, len(self)-1)
        return Array.arg_max(self)

    def entropy(self):
        entropy_value = 0
        for prob in self:
            entropy_value -= prob.data * prob.log().data
        return entropy_value

