import numpy as np
from random import random
import time
import math
from enum import Enum
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import simpledialog
import threading


class PolicyGradient(object):
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
        self.reward_list = []
        
        self.end = False
        self.pause = False

        self.model_init()
    
    class Activation(Enum):  
        def RELU(x, alpha):
            return np.maximum(0, x)

        def LEAKY_RELU(x, alpha):
            return np.where(x > 0, x, alpha * x)

        def ELU(x, alpha):
            return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

        def SWISH(x, alpha):
            return x / (1 + np.exp(-x))
    
    class Optimization(Enum):
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

        def NADAM(grad, cache1, cache2, beta1, beta2, lr, epoch):
            c1 = beta1 * cache1 + (1 - beta1) * grad
            c2 = beta2 * cache2 + (1 - beta2) * grad**2
            c1_corrected = c1 / (1 - beta1 ** (epoch + 1) + 1e-8)
            c2_corrected = c2 / (1 - beta2 ** (epoch) + 1e-8)
            m_t = (1 - beta1) * grad / (1 - beta1 ** (epoch + 1))
            grad_b_add = lr * (c1_corrected + beta1 * m_t) / (np.sqrt(c2_corrected) + 1e-8)
            return grad_b_add, c1, c2
    
    class BackPropagation(Enum):
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
            
            self.epoch = epoch
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
                plt.plot(np.arange(self.epoch), self.reward_list, label='Reward')

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
        if action == np.argmax(self.numbers):
            reward = 1
        else:
            reward = -1
        if test:
            print('Action', action)
            print('Total', total)
            print('Reward', reward)
            time.sleep(0.01)
            
        self.counter += 1
        return self.get_state(), reward, self.counter == 100, None
  
    def render(self):
        print(self.numbers, np.sum(self.numbers))


class PolicyGradientUI(tk.Tk):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.config = {}
        self.activation_var = tk.StringVar(value="RELU")  # Add this line
        self.optimization_var = tk.StringVar(value="RMSPROP")  # Add this line
        self.backpropagation_var = tk.StringVar(value="RELU")  # Add this line
        self.policy_gradient = PolicyGradient(self.env, self.config)
        self.create_widgets()
            
    def create_widgets(self):
        epochs_label = ttk.Label(self, text="Number of Epochs:")
        epochs_label.grid(row=1, column=0, padx=10, pady=5)
        self.epochs_entry = ttk.Entry(self)
        self.epochs_entry.grid(row=1, column=1, padx=10, pady=5)
        self.epochs_entry.insert(0, "10000")

        cmax_label = ttk.Label(self, text="Cmax:")
        cmax_label.grid(row=2, column=0, padx=10, pady=5)
        self.cmax_entry = ttk.Entry(self)
        self.cmax_entry.grid(row=2, column=1, padx=10, pady=5)
        self.cmax_entry.insert(0, "100")

        batch_size_label = ttk.Label(self, text="Batch Size:")
        batch_size_label.grid(row=3, column=0, padx=10, pady=5)
        self.batch_size_entry = ttk.Entry(self)
        self.batch_size_entry.grid(row=3, column=1, padx=10, pady=5)
        self.batch_size_entry.insert(0, "10")
        
        train_button = ttk.Button(self, text="Train", command=self.train)
        train_button.grid(row=4, column=0, columnspan=2, pady=10)

        reset_training = ttk.Button(self, text="Reset Train", command=self.reset)
        reset_training.grid(row=5, column=0, columnspan=2, pady=5)
        
        reset_alg_button = ttk.Button(self, text="Reset Algorithm", command=self.show_reset_dialog)
        reset_alg_button.grid(row=6, column=0, columnspan=2, pady=5)

        view_settings_button = ttk.Button(self, text="View Settings", command=self.view_settings)
        view_settings_button.grid(row=7, column=0, columnspan=2, pady=5)

        pause_resume_button = ttk.Button(self, text="Pause/Resume", command=self.pause_resume)
        pause_resume_button.grid(row=8, column=0, columnspan=2, pady=5)
        
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Reward')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().grid(row=0, column=2, rowspan=9, padx=10, pady=10)
        
        self.progress_var = tk.DoubleVar()
        progress_label = ttk.Label(self, text="Training Progress:")
        progress_label.grid(row=10, column=0, columnspan=2, pady=5)
        self.progress_bar = ttk.Progressbar(self, variable=self.progress_var, length=200, mode="determinate")
        self.progress_bar.grid(row=11, column=0, columnspan=2, pady=5)
        
        self.training_thread = None
    
    def show_reset_dialog(self):
        popup = tk.Toplevel(self)
        popup.title("Reset Algorithm")

        learning_rate_label = ttk.Label(popup, text="Learning Rate:")
        learning_rate_label.grid(row=0, column=0, padx=10, pady=5)
        learning_rate_entry = ttk.Entry(popup)
        learning_rate_entry.grid(row=0, column=1, padx=10, pady=5)
        learning_rate_entry.insert(0, str(self.policy_gradient.learning_rate))

        hidden_layers_label = ttk.Label(popup, text="Hidden Layers:")
        hidden_layers_label.grid(row=1, column=0, padx=10, pady=5)
        hidden_layers_entry = ttk.Entry(popup)
        hidden_layers_entry.grid(row=1, column=1, padx=10, pady=5)
        hidden_layers_entry.insert(0, str(self.policy_gradient.hidden_layers))

        alpha_label = ttk.Label(popup, text="Alpha:")
        alpha_label.grid(row=2, column=0, padx=10, pady=5)
        alpha_entry = ttk.Entry(popup)
        alpha_entry.grid(row=2, column=1, padx=10, pady=5)
        alpha_entry.insert(0, str(self.policy_gradient.alpha))

        discount_label = ttk.Label(popup, text="Discount:")
        discount_label.grid(row=3, column=0, padx=10, pady=5)
        discount_entry = ttk.Entry(popup)
        discount_entry.grid(row=3, column=1, padx=10, pady=5)
        discount_entry.insert(0, str(self.policy_gradient.discount))

        beta1_label = ttk.Label(popup, text="Beta1:")
        beta1_label.grid(row=4, column=0, padx=10, pady=5)
        beta1_entry = ttk.Entry(popup)
        beta1_entry.grid(row=4, column=1, padx=10, pady=5)
        beta1_entry.insert(0, str(self.policy_gradient.beta1))

        beta2_label = ttk.Label(popup, text="Beta2:")
        beta2_label.grid(row=5, column=0, padx=10, pady=5)
        beta2_entry = ttk.Entry(popup)
        beta2_entry.grid(row=5, column=1, padx=10, pady=5)
        beta2_entry.insert(0, str(self.policy_gradient.beta2))

        exploration_rate_label = ttk.Label(popup, text="Exploration Rate:")
        exploration_rate_label.grid(row=6, column=0, padx=10, pady=5)
        exploration_rate_entry = ttk.Entry(popup)
        exploration_rate_entry.grid(row=6, column=1, padx=10, pady=5)
        exploration_rate_entry.insert(0, str(self.policy_gradient.exploration_rate))

        activation_label = ttk.Label(popup, text="Activation Function:")
        activation_label.grid(row=7, column=0, padx=10, pady=5)
        activation_var = tk.StringVar(value=self.policy_gradient.activation)
        activation_dropdown = ttk.Combobox(popup, textvariable=activation_var, values=["RELU", "LEAKY_RELU", "ELU", "SWISH"])
        activation_dropdown.grid(row=7, column=1, padx=10, pady=5)

        optimization_label = ttk.Label(popup, text="Optimization Function:")
        optimization_label.grid(row=8, column=0, padx=10, pady=5)
        optimization_var = tk.StringVar(value=self.policy_gradient.optimization)
        optimization_dropdown = ttk.Combobox(popup, textvariable=optimization_var, values=["RMSPROP", "SGD", "ADAM", "NADAM"])
        optimization_dropdown.grid(row=8, column=1, padx=10, pady=5)

        backpropagation_label = ttk.Label(popup, text="Backpropagation Function:")
        backpropagation_label.grid(row=9, column=0, padx=10, pady=5)
        backpropagation_var = tk.StringVar(value=self.policy_gradient.backpropagation)
        backpropagation_dropdown = ttk.Combobox(popup, textvariable=backpropagation_var, values=["RELU", "LEAKYRELU", "ELU", "SWISH"])
        backpropagation_dropdown.grid(row=9, column=1, padx=10, pady=5)

        apply_button = ttk.Button(popup, text="Apply", command=lambda: self.apply_config_changes(
            popup, learning_rate_entry.get(), hidden_layers_entry.get(),
            alpha_entry.get(), discount_entry.get(), beta1_entry.get(),
            beta2_entry.get(), exploration_rate_entry.get(),
            activation_var.get(), optimization_var.get(), backpropagation_var.get()
        ))
        apply_button.grid(row=10, column=0, columnspan=2, pady=10)

    def apply_config_changes(self, popup, learning_rate, hidden_layers, alpha, discount, beta1, beta2, exploration_rate,
                             activation, optimization, backpropagation):
        try:
            self.policy_gradient.learning_rate = float(learning_rate)
            self.policy_gradient.hidden_layers = int(hidden_layers)
            self.policy_gradient.alpha = float(alpha)
            self.policy_gradient.discount = float(discount)
            self.policy_gradient.beta1 = float(beta1)
            self.policy_gradient.beta2 = float(beta2)
            self.policy_gradient.exploration_rate = float(exploration_rate)
            self.policy_gradient.activation = getattr(self.policy_gradient.Activation, activation)
            self.policy_gradient.optimization = getattr(self.policy_gradient.Optimization, optimization)
            self.policy_gradient.backpropagation = getattr(self.policy_gradient.BackPropagation, backpropagation)

            popup.destroy()
        except:
            pass

    def view_settings(self):
        popup = tk.Toplevel(self)
        popup.title("View Settings")

        settings_text = f"Learning Rate: {self.policy_gradient.learning_rate}\n" \
                f"Hidden Layers: {self.policy_gradient.hidden_layers}\n" \
                f"Alpha: {self.policy_gradient.alpha}\n" \
                f"Discount: {self.policy_gradient.discount}\n" \
                f"Beta1: {self.policy_gradient.beta1}\n" \
                f"Beta2: {self.policy_gradient.beta2}\n" \
                f"Exploration Rate: {self.policy_gradient.exploration_rate}\n" \
                f"Activation Function: {self.policy_gradient.activation.__name__}\n" \
                f"Optimization Function: {self.policy_gradient.optimization.__name__}\n" \
                f"Backpropagation Function: {self.policy_gradient.backpropagation.__name__}"


        settings_label = tk.Label(popup, text=settings_text)
        settings_label.pack(padx=10, pady=10)

        close_button = ttk.Button(popup, text="Close", command=popup.destroy)
        close_button.pack(pady=10)
    
    def pause_resume(self):
        if self.policy_gradient.pause:
            self.policy_gradient.pause = False
        else:
            self.policy_gradient.pause = True
    
    def reset_alg(self):
        self.reset()

        activation = getattr(self.policy_gradient.Activation, self.activation_var.get())
        optimization = getattr(self.policy_gradient.Optimization, self.optimization_var.get())
        backpropagation = getattr(self.policy_gradient.BackPropagation, self.backpropagation_var.get())

        config = {
            'activation': activation,
            'optimization': optimization,
            'backpropagation': backpropagation,
            'learning_rate': 0.01,
            'hidden_layers': 20,
        }

        self.policy_gradient.__init__(self.env, config)
    
    def train(self):
        self.reset()
        
        activation = getattr(self.policy_gradient.Activation, self.activation_var.get())
        optimization = getattr(self.policy_gradient.Optimization, self.optimization_var.get())
        backpropagation = getattr(self.policy_gradient.BackPropagation, self.backpropagation_var.get())

        config = {
            'activation': activation,
            'optimization': optimization,
            'backpropagation': backpropagation,
            'learning_rate': 0.01,
            'hidden_layers': 20,
        }

        epochs = int(self.epochs_entry.get())
        cmax = int(self.cmax_entry.get())
        batch_size = int(self.batch_size_entry.get())

        self.policy_gradient.config = config
        
        self.policy_gradient.reward_list = []

        self.policy_gradient.end = False
        
        self.update_graph(epochs)
        self.training_thread = threading.Thread(target=self.policy_gradient.train, args=(epochs, batch_size, cmax))
        self.training_thread.start()
    
    def reset(self):
        self.policy_gradient.pause = False
        self.policy_gradient.end = True
        if self.training_thread:
            try:
                self.training_thread.stop()
                self.training_thread.join()
            except:
                pass
        self.policy_gradient.epoch = 0
        time.sleep(0.5)
        self.ax.clear()
        self.canvas.draw()
    
    def update_graph(self, epochs):
        self.stop_training_flag = False
        
        def update():
            while not self.policy_gradient.end:
                try:
                    self.ax.clear()
                    self.ax.plot(np.arange(self.policy_gradient.epoch), self.policy_gradient.reward_list, label='Reward')
                    self.ax.set_xlabel('Epoch')
                    self.ax.set_ylabel('Reward')
                    self.canvas.draw()
                except:
                    pass

                progress_value = (len(self.policy_gradient.reward_list) / epochs) * 100
                self.progress_var.set(progress_value)
                
                time.sleep(0.25)
            
            self.progress_var.set(0)
            self.training_thread = None
        
        self.training_thread = threading.Thread(target=update)
        self.training_thread.start()

if __name__ == "__main__":
    test_env = TestEnv()
    app = PolicyGradientUI(test_env)
    app.mainloop()

