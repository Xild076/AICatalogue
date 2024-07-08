from engine import value
import numpy as np

class Categorical:
    def __init__(self, probs=None, logits=None):
        if probs is not None:
            self.probs = probs
            self.logits = self._prob_logit()
        elif logits is not None:
            self.logits = logits
            self.probs = self._logit_prob()
        else:
            raise ValueError("Either probs or logits must be provided")
        
    def _prob_logit(self):
        return self.probs.log()
    
    def _logit_prob(self):
        return self.logits.softmax()
    
    def log_prob(self, categories):
        if self.probs.data.ndim == 1:
            category_prob = self.probs.data[categories]
            out = value(np.log(category_prob + 1e-8), (self.probs,), 'lp_categorical')
            
            def _backward():
                self.probs.grad[categories] += (1 / (category_prob + 1e-8)) * out.grad
                self.probs.grad -= 1
            out._backward = _backward
            
            return out
        else:
            batch_size = self.probs.data.shape[0]
            out = value(np.zeros(batch_size), (self.probs,), 'lp_categorical')
            for i in range(batch_size):
                category_prob = self.probs.data[i, int(categories[i])]
                out.data[int(i)] = np.log(category_prob + 1e-8)
            
            def _backward():
                for i in range(batch_size):
                    category_prob = self.probs.data[i, int(categories[i])]
                    self.probs.grad[i, int(categories[i])] += (1 / (category_prob + 1e-8)) * out.grad[i]
                    self.probs.grad -= 1
            out._backward = _backward
            
            return out
    
    def sample(self):
        if self.probs.data.ndim == 1:
            samples = np.random.choice(len(self.probs.data), p=self.probs.data)
        else:
            samples = np.array([np.random.choice(len(self.probs.data[i]), p=self.probs.data[i]) for i in range(self.probs.data.shape[0])])
        return samples
    
    def __repr__(self):
        return f"Categorical(probs={self.probs.data.tolist()}, logits={self.logits.data.tolist()})"
