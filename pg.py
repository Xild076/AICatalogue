import numpy as np
from engine import value
from nn import MLP
from optim import ADAM, SGD, RMSProp
from dist import Categorical
import gym
import matplotlib.pyplot as plt

class PolicyNet(MLP):
    def __init__(self, layers, activ, init_method):
        super().__init__(layers, activ, init_method)
    
    def forward(self, x):
        x = value(x) if not isinstance(x, value) else x
        return self(x).softmax()

class VPG:
    def __init__(self, env, policy_net, optim_class, lr):
        self.env = env
        self.policy_net = policy_net
        self.optim = optim_class(policy_net.parameters(), lr)
    
    def discount(self, rewards, gamma):
        discounted_rewards = np.zeros_like(rewards, dtype=np.float64)
        cumulative = 0.0
        for i in reversed(range(len(rewards))):
            cumulative = cumulative * gamma + rewards[i]
            discounted_rewards[i] = cumulative
        return value(discounted_rewards)
    
    def train(self, epochs):
        returns = []
        
        for epoch in range(epochs):
            rewards = []
            actions = []
            states = []
            
            state, _ = self.env.reset()
            counter = 0
            while True:
                counter += 1
                state = value(state)
                probs = self.policy_net.forward(state)
                sampler = Categorical(probs=probs)
                action = sampler.sample()
                n_s, r, done, _, _ = self.env.step(action)
                env.render()
                
                states.append(state.data)
                actions.append(action)
                rewards.append(r)
                
                state = n_s
                if done or counter == 500:
                    break
            
            R = self.discount(rewards, 0.99)
            states = value(states)
            
            probs = self.policy_net.forward(states)
            sampler = Categorical(probs=probs)
            lp = -sampler.log_prob(actions)
            loss = (lp * R).sum()
            
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            
            returns.append(sum(rewards))
            print(f'Episode {epoch} | Reward {sum(rewards)}')
            
        return returns


lr = 0.001
env = gym.make('CartPole-v1', render_mode='human')
#env = gym.make('CartPole-v1')
env.metadata['render_fps'] = 640
policy_net = PolicyNet(layers=[4, 16, 2], activ=['relu', ''], init_method='xavier')
vpg_instance = VPG(env, policy_net, RMSProp, lr)

score = vpg_instance.train(5000)
x = [i for i in range(len(score))]

z = np.polyfit(x, score, 25)
p = np.poly1d(z)

plt.plot(x, score)
plt.plot(x, p(x))
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Training Progress of VPG on CartPole-v1')
plt.show()
