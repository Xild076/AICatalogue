import numpy as np
import random
import math
import Categorical
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.animation import FuncAnimation
import threading


class RewardEnv(object):
    def __init__(self):
        pass
    
    def step(self):
        pass
    
    def reset(self):
        pass
    
    def get_state(self):
        pass
    

class LossEnv(object):
    def __init__(self):
        pass
    
    def get_train(self):
        pass


class HelicopterEnv(RewardEnv):
    def __init__(self):
        super().__init__()
        
        self._r = [random.randint(-50, -25), random.randint(-50, -25), 0]
        self._v = [0, 0, 0]
        
        self._g = [random.randint(25, 50), random.randint(25, 50), random.randint(0, 20)]
        
        self.wind = [random.random() - 0.5, random.random() - 0.5, random.random() - 0.5]
        
        self._lift = 0
        self._direction = 0
        self._tilt = 0
        
        self.tick = 0.5
        self._counter = 0
    
    def _calc_a(self):
        forward_force = self._lift * math.sin(-self._tilt)
        z_a = self._lift * math.cos(-self._tilt)
        
        x_a = forward_force * math.cos(self._direction)
        y_a = forward_force * math.sin(self._direction)
        
        x_a += self.wind[0]
        y_a += self.wind[1]
        z_a += self.wind[2]
        
        z_a -= 9.81
        
        if self._v[2] < 0:
            z_a += 0.5 * 1.293 * self._v[2] ** 2 * 0.3
    
        return (x_a, y_a, z_a)
    
    def _calc_v(self):
        a = self._calc_a()
        if self._v != (0, 0, 0) and self._r[2] != 0:
            self._v[0] += a[0] * self.tick
            self._v[1] += a[1] * self.tick
            self._v[2] += a[2] * self.tick
        else:
            if a[2] > 0:
                self._v[2] += a[2] * self.tick
    
    def _calc_l(self):
        death = False
        
        self._r[0] += self._v[0] * self.tick
        self._r[1] += self._v[1] * self.tick
        
        z_v = self._v[2]
        self._r[2] = z_v * self.tick
        if self._r[2] < 0:
            if z_v < -1:
                death = True
            self._r[2] = 0
        
        return death

    def calc_dist(self):
        distance = math.sqrt((self._r[0] - self._g[0])**2 +
                         (self._r[1] - self._g[1])**2 +
                         (self._r[2] - self._g[2])**2)
        
        return distance

    def get_state(self):
        return np.array([self._r[0], self._r[1], self._r[2], self._v[0], self._v[1], self._v[2], self._lift, self._tilt, self._direction, self.wind[0], self.wind[1], self.wind[2], self._g[0], self._g[1], self._g[2]])
    
    def step(self, actions, debug=False):
        done = False
        
        self._lift = self._limit_float(self._lift + actions[0], 0, 20)
        self._direction += self._limit_float(actions[1], -1.04719755, 1.04719755)
        self._tilt = self._limit_float(self._tilt + actions[2], -1.04719755, 1.04719755)
        
        self.wind[0] = self._limit_float(random.random() + self.wind[0], -1.5, 1.5)
        self.wind[1] = self._limit_float(random.random() + self.wind[1], -1.5, 1.5)
        self.wind[2] = self._limit_float(random.random() + self.wind[2], -0.5, 0.5)
        
        before_dist = self.calc_dist()
        
        self._calc_v()
        death = self._calc_l()
        
        after_dist = self.calc_dist()
        
        if debug:
            print("act", actions)
            print("lift", self._lift)
            print("direction", self._direction)
            print("wind", self.wind)
            print("Before dist", before_dist)
            print("after dist", after_dist)
            print("ded", death)
            print("loc", self._r)
            print("vel", self._v)
            print("goal", self._g)
        
        reward = 0
        
        if before_dist > after_dist:
            reward += 20
            if debug:
                print("dist less")
        else:
            reward -= 25
        
        if before_dist != after_dist:
            if debug:
                print("moved")
            reward += 5
        else:
            reward -= 10
        
        if death:
            reward -= 4000
            done = True
        if after_dist <= 2:
            reward += 1000
            done = True
        if self._counter == 200:
            reward = -10000
            done = True
        
        if debug:
            print("reward", reward)
            input()
        
        self._counter += 1
            
        return self.get_state(), reward, done, None
        
    def _limit_float(self, val, min_v, max_v):
        return max(min(val, max_v), min_v)
    
    def reset(self):
        self.__init__()
        return self.get_state()
    


class Map2d(RewardEnv):
    def __init__(self):
        super().__init__()
        self.goal = [random.randint(5, 10), random.randint(5, 10)]
        self.loc = [random.randint(0, 5), random.randint(0, 5)]
        self.counter = 0
    
    def step(self, action):
        done = False
        reward = 0
                
        old_dist = math.sqrt((self.goal[0] - self.loc[0]) ** 2 + (self.goal[1] - self.loc[1]) ** 2)
        
        if action == 0:
            self.loc[0] += 1
        if action == 1:
            self.loc[0] -= 1
        if action == 2:
            self.loc[1] += 1
        if action == 3:
            self.loc[1] -= 1
        
        new_dist = math.sqrt((self.goal[0] - self.loc[0]) ** 2 + (self.goal[1] - self.loc[1]) ** 2)
        
        reward += (old_dist - new_dist) / 10
                
        if self.goal == self.loc:
            reward += 10
            done = True
        
        if self.counter == 50:
            reward -= 10
            done = True
        
        self.counter += 1
        
        return self.get_state(), reward, done, None
    
    def get_state(self):
        return np.array([self.goal[0], self.goal[1], self.loc[0], self.loc[1]])
    
    def reset(self):
        self.__init__()
        return self.get_state()


h = Map2d()
m = Categorical.Categorical({'lr': 0.001, 'optm': 'rmsprop', 'type': 'reward', 'lt': 'mse', 'cont': False}, [4, 200, 4], ['', ''])
o, _ = m.train({'epochs': 10000, 'batch': 10, 'env': Map2d()})
m.visualize(o, _)

done = False
old_state = 0
state = h.reset()
reward_sum = 0
while not done:
    passes = m.forward(state - old_state)
    print(passes[-1])
    act, act_grad = m.sample_action(passes.pop(-1))
    print(act)
    state, reward, done, _ = h.step(act)
    reward_sum += reward

print(reward_sum)