import numpy as np
from random import random
import time
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import threading


class PolicyGradient(object):
    """
    A class for implementing a Policy Gradient algorithm.

    Parameters
    ----------
    env : gym.Env
        The environment to be trained on.
    config : dict
        A dictionary containing the configuration parameters for the algorithm.

    Attributes
    ----------
    env : gym.Env
        The environment to be trained on.
    num_states : int
        The number of states in the environment.
    num_actions : int
        The number of actions available in the environment.
    learning_rate : float
        The learning rate used for updating the policy parameters.
    hidden_layers : int
        The number of hidden layers in the policy network.
    activation : function
        The activation function used in the policy network.
    optimization : function
        The optimization function used for updating the policy parameters.
    backpropagation : function
        The backpropagation function used in the policy network.
    alpha : float
        The alpha parameter used in the activation functions.
    discount : float
        The discount factor used for calculating the returns.
    beta1 : float
        The beta1 parameter used in the optimization functions.
    beta2 : float
        The beta2 parameter used in the optimization functions.
    exploration_rate : float
        The exploration rate used for exploring the action space.
    continuous : bool
        A flag indicating whether the environment is continuous or not.
    reward_list : list
        A list containing the rewards obtained during training.

    Methods
    -------
    model_init()
        Initializes the policy network parameters.
    softmax(x)
        Computes the softmax function.
    policy_forward(state)
        Computes the output of the policy network for a given state.
    discount_rewards(rewards)
        Discounts the rewards according to the discount factor.
    policy_backward(state_change, hid, gprob)
        Computes the gradient of the policy loss with respect to the policy parameters.
    train(epochs, batch_size, count_max)
        Trains the policy for a given number of epochs.
    render()
        Renders the training progress.
    train_render(episodes, batch_size, cmax)
        Trains the policy and renders the training progress.
    """
    def __init__(self, env, config):
        self.env = env

        self.num_states = env.observation_space
        self.num_actions = env.action_space
        self.learning_rate = config.get('learning_rate', 0.01)
        self.hidden_layers = config.get('hidden_layers', 20)
        self.activation = config.get('activation', self.Activation.RELU)
        self.optimization = config.get('optimization', self.Optimization.RMSPROP)
        self.backpropagation = config.get('backpropagation', self.BackPropagation.RELU)
        self.alpha = config.get('alpha', 0.99)
        self.discount = config.get('discount', 0.99)
        self.beta1 = config.get('beta1', 0.9)
        self.beta2 = config.get('beta2', 0.999)
        self.exploration_rate = config.get('exploration_rate', 0.02)
        self.continuous = config.get('continuous', False)
        self.reward_list = []
        
        self.end = False
        self.pause = False

        self.model_init()
    
    class Activation():  
        def RELU(x, alpha):
            return np.maximum(0, x)

        def LEAKY_RELU(x, alpha):
            return np.where(x > 0, x, alpha * x)

        def ELU(x, alpha):
            return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

        def SWISH(x, alpha):
            return x / (1 + np.exp(-x))
    
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
    
    class BackPropagation():
        def RELU(dh, hid, alpha):
            dh[hid <= 0] = 0
            return dh
        
        def LEAKYRELU(dh, hid, alpha):
            dh[hid <= 0] = alpha * dh[hid <= 0]
            return dh
        
        def ELU(dh, hid, alpha):
            dh[hid <= 0] = dh[hid <= 0] * (hid[hid <= 0] + alpha)
            return dh
        
        def SWISH(dh, hid, alpha):
            swish_derivative = (1 + np.exp(-hid) + hid * np.exp(-hid)) / (1 + np.exp(-hid))**2
            dh = dh * swish_derivative
            return dh
            
    def model_init(self):
        self.model = {
            1: np.random.randn(self.hidden_layers, self.num_states) / np.sqrt(self.num_actions) * self.learning_rate,
            2: np.random.randn(self.num_actions, self.hidden_layers) / np.sqrt(self.hidden_layers) * self.learning_rate,
        }
    
    def softmax(self, x):
        x = x.astype(np.float64)
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def policy_forward(self, state):
        hid = np.dot(self.model[1], state)
        hid = self.activation(hid, self.alpha)
        logp = np.dot(self.model[2], hid)
        prob = self.softmax(logp)
        return prob, hid
    
    def discount_rewards(self, rewards):
        discounted_reward = np.zeros_like(rewards, dtype=np.float64)
        total_rewards = 0
        for t in reversed(range(0, len(rewards))):
            total_rewards = total_rewards * self.discount + rewards[t]
            discounted_reward[t] = total_rewards
        return discounted_reward
    
    def policy_backward(self, state_change, hid, gprob):
        dW2 = np.dot(hid.T, gprob).T
        dh = np.dot(gprob, self.model[2])
        dh = self.backpropagation(dh, hid, self.alpha)
        dW1 = np.dot(dh.T, state_change)
        return {1: dW1, 2: dW2}

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

            sstate, shidden, sgrads, srewards = [], [], [], []

            state = self.env.reset()
            old_state = 0
            done = False
            counter = 0
            reward_sum = 0

            while not done and counter < count_max:
                counter += 1
                calc_state = state - old_state
                if self.continuous:
                    old_state = state

                prob, hid = self.policy_forward(calc_state)
                
                if random() > e_rate:
                    action = np.random.choice(np.arange(self.num_actions), p=prob)
                else:
                    action = np.random.choice(np.arange(self.num_actions))
                e_rate -= self.exploration_rate / (epoch + 1)

                aoh = np.zeros(self.num_actions)
                aoh[action] = 1

                sstate.append(calc_state)
                shidden.append(hid)
                sgrads.append(aoh - prob)

                state, reward, done, _ = self.env.step(action)
                srewards.append(reward)

                reward_sum += reward

            vstate = np.vstack(sstate)
            vhidden = np.vstack(shidden)
            vgrads = np.vstack(sgrads)
            vrewards = np.vstack(srewards)

            discounted_vrew = self.discount_rewards(vrewards)
            discounted_vrew -= (np.mean(discounted_vrew)).astype(np.float64)
            discounted_vrew /= ((np.std(discounted_vrew)).astype(np.float64) + 1e-8)

            vgrads *= discounted_vrew
            grad = self.policy_backward(vstate, vhidden, vgrads)
            for k in self.model:
                grad_buffer[k] += grad[k]

            if epoch % batch_size == 0:
                for k, v in self.model.items():
                    g = grad_buffer[k]
                    grad_add, self.cache_1[k], self.cache_2[k] = self.optimization(g, self.cache_1[k], self.cache_2[k],
                                                                                self.beta1, self.beta2, self.learning_rate, epoch)
                    grad_buffer[k] = np.zeros_like(v) 
                    self.model[k] += grad_add

            self.reward_list.append(reward_sum)
            

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
