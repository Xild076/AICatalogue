from engine import value
from random import random, randint
from nn import MLP, Neuron, Layer
import copy


class Categorical:
    def __init__(self, probs=None, logits=None):
        if probs:
            self.probs = probs
            self.logits = self._prob_logit()
        if logits:
            self.logits = logits
            self.probs = self._logit_prob()
        
    def _prob_logit(self):
        return self.probs.log()
    
    def _logit_prob(self):
        return self.logits.softmax()
    
    def log_prob(self, values):
        def _single_lp(probs, v):
            return probs[v].log()

        if isinstance(values, list):
            multi_probs = value.split(self.probs)
            lps = [_single_lp(prob, v) for prob, v in zip(multi_probs, values)]
            return value.combine(lps)
        return _single_lp(self.probs, values)
    
    def __repr__(self) -> str:
        return f'Categorical(logits={self.logits}, probs={self.probs})'

    def sample(self, rand=0.0):
        if len(self.probs.shape) == 1:
            return self._single_sample(self.probs, rand)
        return self._multi_sample(self.probs, rand)
    
    def _single_sample(self, probs, rand):
        if random() <= rand:
            return randint(0, len(probs) - 1)
        return probs.argmax()
    
    def _multi_sample(self, probs, rand):
        multi_probs = value.split(probs)
        return [self._single_sample(prob, rand) for prob in multi_probs]
