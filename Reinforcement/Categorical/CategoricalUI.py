import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import simpledialog
import Reinforcement.Categorical.Categorical as Categorical
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import time
import numpy as np


class PolicyGradientUI(tk.Tk):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.config = {}
        self.activation_var = tk.StringVar(value="RELU")  # Add this line
        self.optimization_var = tk.StringVar(value="RMSPROP")  # Add this line
        self.backpropagation_var = tk.StringVar(value="RELU")  # Add this line
        self.policy_gradient = Categorical.PolicyGradient(self.env, self.config)
        self.create_widgets()
            
    def create_widgets(self):
        self.title("Policy Gradient UI")
        
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
            config = {
                'learning_rate': float(learning_rate),
                'hidden_layers': int(hidden_layers),
                'alpha': float(alpha),
                'beta1': float(beta1),
                'beta2': float(beta2),
                'exploration_rate': float(exploration_rate),
                'discount': float(discount),
                'activation': getattr(self.policy_gradient.Activation, activation),
                'optimization': getattr(self.policy_gradient.Optimization, optimization),
                'backpropagation': getattr(self.policy_gradient.BackPropagation, backpropagation)    
            }
            
            self.policy_gradient = Categorical.PolicyGradient(self.env, config)
            
            self.reset()

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