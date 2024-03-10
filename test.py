import numpy as np
import gym
import seaborn as sns
from random import random
import matplotlib.pyplot as plt

class DiagonalGaussianPolicy():
    def __init__(self, env, lr=1e-2):
        self.N = env.observation_space
        self.M = env.action_space

        self.weights_1 = np.random.randn(self.N, 64) / np.sqrt(64) * lr
        self.weights_2 = np.random.randn(64, 64) / np.sqrt(64) * lr
        self.weights_3 = np.random.randn(64, self.M) / np.sqrt(self.M) * lr

        self.bias_1 = np.zeros(64)
        self.bias_2 = np.zeros(64)
        self.bias_3 = np.zeros(self.M)

        self.log_sigma = np.ones(self.M, dtype=np.float64)
        self.lr = lr

    def relu(self, x):
        return np.maximum(0, x)

    def pi(self, s_t):
        s_t = s_t[0]
        h1 = np.dot(s_t, self.weights_1) + self.bias_1
        h1 = self.relu(h1)

        h2 = np.dot(h1, self.weights_2) + self.bias_2
        h2 = self.relu(h2)

        mu = np.dot(h2, self.weights_3) + self.bias_3
        sigma = np.exp(self.log_sigma)
        pi = mu + sigma * np.random.randn(self.M)
        return pi

def calculate_returns(rewards, dones, gamma=0.99):
    discounted_reward = np.zeros_like(rewards, dtype=np.float64)
    total_rewards = 0
    for t in reversed(range(0, len(rewards))):
        total_rewards = total_rewards * gamma + rewards[t]
        discounted_reward[t] = total_rewards
    return discounted_reward

def reinforce(env, agent, gamma=0.99, epochs=100, T=1000):
    totals = []

    for epoch in range(epochs):
        s_t = env.reset()

        states = np.empty((T, env.observation_space))
        actions = np.empty((T, env.action_space))
        rewards = np.empty((T,))
        dones = np.empty((T,))

        for t in range(T):
            a_t = agent.pi(s_t)
            print(a_t)
            s_t_next, r_t, d_t, _ = env.step(a_t)

            states[t] = s_t
            actions[t] = a_t
            rewards[t] = r_t
            dones[t] = d_t

            s_t = s_t_next

        returns = calculate_returns(rewards, dones, gamma)

        for t in range(T):
            s_t = states[t]
            a_t = actions[t]
            r_t = returns[t]

            h1 = np.dot(s_t, agent.weights_1) + agent.bias_1
            h1 = agent.relu(h1)

            h2 = np.dot(h1, agent.weights_2) + agent.bias_2
            h2 = agent.relu(h2)

            mu = np.dot(h2, agent.weights_3) + agent.bias_3
            sigma = np.exp(agent.log_sigma)

            grad_log_pi = (a_t - mu) / (sigma**2)
            grad_log_pi_wrt_mu = grad_log_pi

            grad_log_pi_wrt_sigma = ((a_t - mu)**2 / sigma**2) - 1
            grad_log_pi_wrt_sigma /= 2 * sigma

            grad_log_pi_wrt_h2 = np.dot(grad_log_pi_wrt_sigma, agent.weights_3.T)
            grad_log_pi_wrt_h2[h2 <= 0] = 0

            grad_log_pi_wrt_h1 = np.dot(grad_log_pi_wrt_h2, agent.weights_2.T)
            grad_log_pi_wrt_h1[h1 <= 0] = 0

            grad_log_pi_wrt_weights_1 = np.outer(s_t, grad_log_pi_wrt_h1)
            grad_log_pi_wrt_weights_2 = np.outer(h1, grad_log_pi_wrt_h2)
            grad_log_pi_wrt_weights_3 = np.outer(h2, grad_log_pi_wrt_sigma)

            agent.weights_1 += agent.lr * r_t * grad_log_pi_wrt_weights_1
            agent.weights_2 += agent.lr * r_t * grad_log_pi_wrt_weights_2
            agent.weights_3 += agent.lr * r_t * grad_log_pi_wrt_weights_3

            agent.bias_1 += agent.lr * r_t * grad_log_pi_wrt_h1
            agent.bias_2 += agent.lr * r_t * grad_log_pi_wrt_h2
            agent.bias_3 += agent.lr * r_t * grad_log_pi_wrt_sigma

        totals.append(rewards.sum() / dones.sum())
        print(f'{epoch}/{epochs}:{totals[-1]}\r', end='')

    sns.lineplot(x=range(len(totals)), y=totals)
    plt.show()


class TestEnv(object):
    def __init__(self):
        self.observation_space = 3
        self.action_space = 3
        self.numbers = [50*random(), 50*random(), 50*random()]
        self.counter = 0

    def reset(self):
        self.numbers = [50*random(), 50*random(), 50*random()]
        self.counter = 0
        return self.get_state()

    def get_state(self):
        self.numbers = [50*random(), 50*random(), 50*random()]
        return np.array(self.numbers)

    def step(self, action, test=False):
        total = np.sum(self.numbers)
        reward = 10 - abs(self.numbers[0] - action[0]) - abs(self.numbers[1] - action[1]) - abs(self.numbers[2] - action[2])
        if test:
            print('Action', action)
            print('Total', total)
            print('Reward', reward)
            time.sleep(0.01)

        self.counter += 1
        return self.get_state(), reward, self.counter == 100, None

    def render(self):
        print(self.numbers, np.sum(self.numbers))


env = TestEnv()
agent = DiagonalGaussianPolicy(env, lr=1e-1)
reinforce(env, agent)

