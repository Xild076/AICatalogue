import math
from engine import value
import random


class env:
    def __init__(self):
        self.action_space = None
        self.observation_space = None
    
    def get_state(self) -> value:
        # Return state
        pass
    
    def step(self, a):
        # Return Reward, State, Done, Info
        pass
    
    def reset(self) -> value:
        # Return state
        pass


class jump_env(env):
    def __init__(self):
        self.action_space = 5
        self.observation_space = 2
        self.loc = random.randint(-20, 20)
        self.goal = random.randint(-5, 5)
    
    def get_state(self) -> value:
        return value([self.loc, self.goal])
    
    def step(self, a):
        d = a - 2
        dist_old = abs(self.loc - self.goal)
        self.loc += d
        dist_new = abs(self.loc - self.goal)
        
        r = dist_old - dist_new
        done = self.loc == self.goal
        
        return self.get_state(), r, done
    
    def reset(self):
        self.loc = random.randint(-20, 20)
        self.goal = random.randint(-5, 5)
        
        return self.get_state()