import numpy as np
import gym
import matplotlib.pyplot as plt
from engine import value
from nn import MLP
from optim import ADAM
from dist import Categorical
import time
import functional


class PolicyNet(MLP):
    def __init__(self, layers, activ, init_method):
        super().__init__(layers, activ, init_method)
    
    def forward(self, x):
        x = value(x) if not isinstance(x, value) else x
        return self(x).softmax()

class ValueNet(MLP):
    def __init__(self, layers, activ, init_method):
        super().__init__(layers, activ, init_method)
    
    def forward(self, x):
        x = value(x) if not isinstance(x, value) else x
        return self(x)

class PPO:
    def __init__(self, env, policy_net, value_net, policy_lr, value_lr, gamma=0.99, epsilon=0.2, k=4):
        self.env = env
        self.policy_net = policy_net
        self.value_net = value_net
        self.policy_optimizer = ADAM(policy_net.parameters(), policy_lr)
        self.value_optimizer = ADAM(value_net.parameters(), value_lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.k = k
    
    def discount_rewards(self, rewards, gamma):
        discounted_rewards = np.zeros_like(rewards, dtype=np.float64)
        cumulative = 0.0
        for i in reversed(range(len(rewards))):
            cumulative = cumulative * gamma + rewards[i]
            discounted_rewards[i] = cumulative
        return value(discounted_rewards)
    
    def train(self, epochs, batch_size=64):
        returns = []
        
        for epoch in range(epochs):
            states, actions, rewards, old_probs, values, masks = [], [], [], [], [], []
            state, _ = self.env.reset()
            done = False
            while not done:
                state = value(state)
                probs = self.policy_net.forward(state)
                sampler = Categorical(probs=probs)
                action = sampler.sample()
                log_prob = sampler.log_prob(action)
                value_pred = self.value_net.forward(state)
                
                n_s, r, done, _, _ = self.env.step(action)
                
                states.append(state)
                actions.append(action)
                rewards.append(r)
                old_probs.append(log_prob)
                values.append(value_pred)
                masks.append(1 - done)
                
                state = n_s
            
            returns.append(sum(rewards))
            R = self.discount_rewards(rewards, self.gamma)
            values = value.transform(values)
            advantages = R - values
            
            states = value.transform(states)
            old_probs = value.transform(old_probs)
            returns_val = R
            
            for _ in range(self.k):
                probs = self.policy_net.forward(states)
                sampler = Categorical(probs=probs)
                log_probs = sampler.log_prob(actions)
                ratio = (log_probs - old_probs).exp()
                surr1 = ratio * advantages
                surr2 = ratio.clip(1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
                policy_loss = -surr1.minimum(surr2).mean()
                
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                print(policy_loss)
                print(surr2)
                self.policy_optimizer.step()
                
                value_pred = self.value_net.forward(states).flatten()
                print(value_pred.shape)
                print(returns_val.shape)
                value_loss = functional.mean_squared_error(value_pred, returns_val, 'none')
                
                self.value_optimizer.zero_grad()
                value_loss.backward()
                print(value_loss)
                print(self.value_net.layers[0].neurons[0].weights)
                time.sleep(10)
                self.value_optimizer.step()
            
            print(f'Epoch {epoch} | Reward {sum(rewards)}')
        
        return returns


policy_lr = 0.0003
value_lr = 0.0003
env = gym.make('CartPole-v1')
env.metadata['render_fps'] = 640
policy_net = PolicyNet(layers=[4, 64, 2], activ=['relu', ''], init_method='xavier')
value_net = ValueNet(layers=[4, 64, 1], activ=['relu', ''], init_method='xavier')
ppo_instance = PPO(env, policy_net, value_net, policy_lr, value_lr)

scores = ppo_instance.train(1000)
x = [i for i in range(len(scores))]

plt.plot(x, scores)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Training Progress of PPO on CartPole-v1')
plt.show()