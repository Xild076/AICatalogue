import numpy as np
from random import random
import time
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import threading


class PolicyGradient(object):
    def __init__(self, env, config):
        self.env = env

        self.num_states = env.observation_space
        self.num_actions = env.action_space
        self.learning_rate = config.get('learning_rate', 0.01)
        self.hidden_layers = config.get('hidden_layers', 100)
        self.alpha = config.get('alpha', 0.99)
        self.optimization = config.get('optimization', self.Optimization.RMSPROP)
        self.discount = config.get('discount', 0.99)
        self.beta1 = config.get('beta1', 0.9)
        self.beta2 = config.get('beta2', 0.999)
        self.exploration_rate = config.get('exploration_rate', 0.02)
        self.reward_list = []

        self.end = False
        self.pause = False

        self.model_init()

    class Optimization():
        def RMSPROP(grad, cache1, cache2, beta1, beta2, lr, epoch):
            rc = beta1 * cache1 + (1 - beta1) * grad**2
            return (lr * grad / (np.sqrt(rc) + 1e-8)), rc, 0

        def SGD(grad, cache1, cache2, beta1, beta2, lr, epoch):
            return lr * grad, 0, 0

        def ADAM(grad, cache1, cache2, beta1, beta2, lr, epoch):
            c1 = beta1 * cache1 + (1 - beta1) * grad
            c2 = beta2 * cache2 + (1 - beta2) * grad**2
            c1_corrected = c1 / (1 - beta1 ** (epoch + 1) + 1e-8)
            c2_corrected = c2 / (1 - beta2 ** epoch + 1e-8)
            grad_b_add = lr * c1_corrected / (np.sqrt(c2_corrected) + 1e-8)
            return grad_b_add, c1, c2
        
        def NAG(grad, cache1, cache2, beta1, beta2, lr, epoch):
            m_t = beta1 * cache1 + (1 - beta1) * grad
            c1 = beta1 * cache1 + (1 - beta1) * m_t
            c2 = beta2 * cache2 + (1 - beta2) * grad**2
            c1_corrected = c1 / (1 - beta1 ** (epoch + 1) + 1e-8)
            c2_corrected = c2 / (1 - beta2 ** epoch + 1e-8)
            grad_b_add = lr * (c1_corrected + beta1 * m_t) / (np.sqrt(c2_corrected) + 1e-8)
            return grad_b_add, c1, c2

        def NADAM(grad, cache1, cache2, beta1, beta2, lr, epoch):
            c1 = beta1 * cache1 + (1 - beta1) * grad
            c2 = beta2 * cache2 + (1 - beta2) * grad**2
            c1_corrected = c1 / (1 - beta1 ** (epoch + 1) + 1e-8)
            c2_corrected = c2 / (1 - beta2 ** (epoch) + 1e-8)
            m_t = (1 - beta1) * grad / (1 - beta1 ** (epoch + 1))
            grad_b_add = lr * (c1_corrected + beta1 * m_t) / (np.sqrt(c2_corrected) + 1e-8)
            return grad_b_add, c1, c2
    
    def model_init(self):
        self.model = {
            'mean': np.random.randn(self.num_actions, self.num_states) / np.sqrt(self.num_states) * self.learning_rate,
            'log_std': np.random.randn(self.num_actions, self.num_states) / np.sqrt(self.num_states) * self.learning_rate,
        }

    def sample_action(self, mean, log_std):
        action = mean + np.exp(log_std) * np.random.normal(size=self.num_actions)
        return action

    def policy_forward(self, state):
        mean_actions = np.dot(self.model['mean'], state)
        log_std = np.dot(self.model['log_std'], state)
        std_dev = np.exp(log_std)

        actions = self.sample_action(mean_actions, log_std)

        return actions

    def discount_rewards(self, rewards):
        discounted_reward = np.zeros_like(rewards, dtype=np.float64)
        total_rewards = 0
        for t in reversed(range(0, len(rewards))):
            total_rewards = total_rewards * self.discount + rewards[t]
            discounted_reward[t] = total_rewards
        discounted_reward -= np.min(discounted_reward)
        discounted_reward /= np.std(discounted_reward) + 1e-8
        return discounted_reward

    def policy_backward(self, state_change, gactions, log_std):
        dmean = np.dot(gactions.T, state_change)
        dlog_std = 0.5 * np.dot(np.dot(gactions, np.exp(log_std)).T, state_change)
        return {'mean': dmean, 'log_std': dlog_std}

    def train(self, epochs, batch_size, count_max):
        self.cache_1 = {k: np.zeros_like(v) for k, v in self.model.items()}
        self.cache_2 = {k: np.zeros_like(v) for k, v in self.model.items()}
        e_rate = self.exploration_rate

        self.reward_list = []

        for epoch in range(epochs):

            if self.end:
                break

            while self.pause:
                time.sleep(0.1)

            grad_buffer = {k: np.zeros_like(v) for k, v in self.model.items()}

            sstate, sgrads, srewards = [], [], []

            state = self.env.reset()
            old_state = 0
            done = False
            counter = 0
            reward_sum = 0

            while not done and counter < count_max:
                counter += 1
                calc_state = state - old_state

                actions = self.policy_forward(calc_state)

                sgrads.append(actions)
                sstate.append(calc_state)

                state, reward, done, _ = self.env.step(actions)
                srewards.append(reward)

                reward_sum += reward

            vstate = np.vstack(sstate)
            vgrads = np.vstack(sgrads)
            vrewards = np.vstack(srewards)

            discounted_vrew = self.discount_rewards(vrewards)

            vgrads *= discounted_vrew
            grad = self.policy_backward(vstate, vgrads, self.model['log_std'])
            for k in self.model:
                grad_buffer[k] += grad[k]

            if epoch % batch_size == 0:
                for k, v in self.model.items():
                    g = grad_buffer[k]
                    grad_add, self.cache_1[k], self.cache_2[k] = self.optimization(g, self.cache_1[k], self.cache_2[k],
                                                                                    self.beta1, self.beta2,
                                                                                    self.learning_rate, epoch)
                    grad_buffer[k] = np.zeros_like(v)
                    grad_add = np.clip(grad_add, -1, 1)
                    self.model[k] += grad_add

            self.reward_list.append(reward_sum)
            print(reward_sum)

    def render(self):
        def better_render(mogus):
            plt.cla()
            plt.plot(np.arange(len(self.reward_list)), self.reward_list, label='Reward')

        ani = FuncAnimation(plt.figure(), better_render, interval=2000, cache_frame_data=False)
        plt.tight_layout()
        plt.show()

    def train_render(self, episodes, batch_size, cmax):
        func1 = threading.Thread(target=self.train, args=(episodes, batch_size, cmax))
        func1.start()
        self.render()
        func1.join()


class TestEnv(object):
    def __init__(self):
        self.observation_space = 3
        self.action_space = 3
        self.numbers = [random(), random(), random()]
        self.counter = 0

    def reset(self):
        self.numbers = [random(), random(), random()]
        self.counter = 0
        return self.get_state()

    def get_state(self):
        self.numbers = [random(), random(), random()]
        return np.array(self.numbers)

    def step(self, action, test=False):
        total = np.sum(self.numbers)
        reward = 1 - abs(total - np.sum(action))
        if test:
            print('Action', action)
            print('Total', total)
            print('Reward', reward)
            time.sleep(0.01)

        self.counter += 1
        return self.get_state(), reward, self.counter == 100, None

    def render(self):
        print(self.numbers, np.sum(self.numbers))


pg = PolicyGradient(TestEnv(), {'learning_rate': 0.00001})
pg.train_render(10000, 10, 100)
