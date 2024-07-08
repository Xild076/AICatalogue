from nn import MLP
from engine import value
from optim import SGD, RMSProp
from dist import Categorical
import random
import numpy as np
import functional
import gym


class FF(MLP):
    def __init__(self, layers, activ, init_method='xavier'):
        super().__init__(layers, activ, init_method)
    
    def forward(self, s):
        s = value(s) if not isinstance(s, value) else s
        return self(s)


class PPO:
    def __init__(self, env, actor_net, value_net, optim_class, lr):
        self.env = env
        self.actor_net = actor_net
        self.value_net = value_net
        self.optim = optim_class(actor_net.parameters() + value_net.parameters(), lr)
        self.gamma = 0.99
        self.kl_coeff = 0.2
        self.vf_coeff = 0.5
    
    def pick_sample_and_logp(self, s):
        with value.no_grad():
            logits = self.actor_net(s)
            probs = logits.softmax()
            c = Categorical(probs=probs)
            a = c.sample()
            lp = -c.log_prob(a)
            return a, logits, lp
    
    def discount(self, rewards, gamma):
        cum_rewards = np.zeros_like(rewards)
        reward_len = len(rewards)
        for j in reversed(range(reward_len)):
            cum_rewards[j] = rewards[j] + (cum_rewards[j+1]*gamma if j+1 < reward_len else 0)
        return cum_rewards
    
    def train(self, epochs):
        returns = []
        
        for epoch in range(epochs):
            states = []
            actions = []
            logits = []
            logprbs = []
            rewards = []
            
            state, _ = self.env.reset()
            counter = 0
            while True:
                counter += 1
                state = value(state)
                states.append(state.data)
                
                a, l, p = self.pick_sample_and_logp(state)
                state, r, done, _, _ = self.env.step(a)
                
                actions.append(a)
                rewards.append(r)
                logits.append(l)
                logprbs.append(p)
                
                if done or counter == 500:
                    break
            
            R = value(self.discount(rewards, self.gamma))
            states = value(states)
            logits_old = value.transform(logits)
            logprbs = value.transform(logprbs)
                        
            value_new = self.value_net(states).flatten()
            logits_new = self.actor_net(states)
                        
            advantages = R - value_new
            
            c_new = Categorical(logits=logits_new)
            logprbs_new = -c_new.log_prob(actions)
            prob_ratio = (logprbs_new - logprbs).exp()
            
            l0 = logits_old - logits_old.amax(axis=1, keepdims=True)
            l1 = logits_new - logits_new.amax(axis=1, keepdims=True)
            e0 = l0.exp()
            e1 = l1.exp()
            e_sum0 = e0.sum(axis=1, keepdims=True)
            e_sum1 = e1.sum(axis=1, keepdims=True)
            p0 = e0 / e_sum0
            kl = (p0 * (l0 - e_sum0.log() - l1 + e_sum1.log())).sum(axis=1, keepdims=True).flatten()
            
            diff = R - value_new
            diff2 = diff ** 2
            mse = diff2.mean()
            
            loss = -advantages * prob_ratio + kl * self.kl_coeff + mse * self.vf_coeff
            
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            
            returns.append(sum(rewards))
            print(f'Episode {epoch} | Reward {sum(rewards)}')
            
            returns.append(sum(rewards))
            
        return returns


lr = 0.001
env = gym.make('CartPole-v1', render_mode='human')
env = gym.make('CartPole-v1')
# env.metadata['render_fps'] = 640
actor_net = FF(layers=[4, 16, 2], activ=['relu', ''], init_method='xavier')
value_net = FF(layers=[4, 16, 1], activ=['relu', ''], init_method='xavier')
vpg_instance = PPO(env, actor_net, value_net, RMSProp, lr)

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
