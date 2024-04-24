import numpy as np
import math
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import utility


class Layer(object):
    """Object class for one layer of a NN"""
    def __init__(self, nin, nout, activation='', lr=0.01):
        """Initializing all parameters with Xavier initialization"""
        self.w = np.random.randn(nout, nin) / np.sqrt(nin) * lr
        self.b = np.random.randn(nout) / np.sqrt(nout) * lr
        self.actv = activation
    
    def __call__(self, x):
        """Forward pass in NN.
        1. 'act = np.dot(self.w, x) + self.b' multiplies all the weights and the inputs and adds bias.
        2. 'self._activate(act) calls the activation function for calculated values"""
        act = np.dot(self.w, x) + self.b
        return self._activate(act)
    
    def _activate(self, x, d=False):
        """Activates the function. There are three current activation functions.
        1. ReLU, rectified linear unit, adds non-linearity.
        2. Sigmoid: squashes the values between [0, 1] and into a sum of 1.
        3. Tanh: squashes the values between [0, 1].
        4. None: No activation."""
        if self.actv == 'relu':
            if d:
                x[x<=0] = 0
                x[x>0] = 1
                return x
            return np.maximum(0, x)
        
        if self.actv == 'sigmoid':
            if d:
                sig = self._activate(x)
                return sig * (1 - sig)
            return 1 / (1 + np.exp(-x))
        
        if self.actv == 'tanh':
            if d:
                th = self._activate(x)
                return 1 - th**2
            return np.tanh(x)
        return x
    
    def parameters(self):
        """Returns the values of the Layer."""
        return self.w, self.b
    
    def __repr__(self):
        """Returns the description of the Layer."""
        return f"{self.actv.upper()} Layer | ({self.w.shape[1]}, {self.w.shape[0]})"
    
    def _backwards(self, p_div, x):
        """Runs a single backwards pass to calculated its derivative relative to the previous one.
        The equation for the output is: o = act(w * x + b)
        The derivative with respect to b is:
        - do/db = 1 * act'(w * x + b)
        The derivative with respect to w is:
        - do/dw = (x) * act'(w * x + b)
        The derivative with respect to the input (x) is:
        - do/dx = (w) * act'(w * x + b)
        do/dx will be the next x for recursive calculations."""
        
        x = np.vstack(x)
        p_div = self._activate(p_div, True).T
        b_grad = np.mean((p_div))
        g_grad = np.vstack(p_div)
        dg = np.dot(g_grad, x.T)
        dn = np.dot(g_grad.T, self.w)
        
        return dg, b_grad, dn
    

class Model(object):
    """Object class for a Policy of a NN"""
    def __init__(self, config: dict, layer: list, act: list):
        """Initialized parameters.
        For config - dict:
        - 'lr': Learning rate - Float
        - 'discount': Reward Discount - Float
        - 'beta': Beta for optimizations (Beta1, Beta2, ..., Betan] - List[Float, Float]
        - 'optm': Optimization method - str ('sgd', 'rmsprop')
        - 'type': Version - str ('loss', 'reward')
        - 'lt': Loss Type (Only when v is loss) - str ('mse', 'mae', '')
        - 'cont': Continuous environment - Boolean (True, False)
        Layers: List of integers for nueron size.
        Act: List of activation types (string), needs to be one less than length layer (last layer doesn't need optimization).
        - '' - No activaiton
        - 'relu' - RELU activation
        - 'sigmoid' - SIGMOID activation
        - 'Tanh' - TANH activation
        """
        
        if len(act) < len(layer) - 1:
            raise ValueError("Activation length needs to be at least one less than length of layer.")
        
        self.lr = config.get('lr', 0.1)
        self.discount = config.get('discount', 0.99)
        
        self.beta = config.get('beta', [0.99, 0.99])
        
        self.optm = config.get('optm', 'sgd')
        self.v = config.get('type', 'loss')
        if self.v == 'loss':
            self.lt = config.get('lt', 'mse')
        if self.v == 'reward':
            self.cont = config.get('cont', True)
        
        self.layers = []
        for i in range(len(layer) - 1):
            self.layers.append(Layer(layer[i], layer[i + 1], act[i], self.lr))        
        
    def __repr__(self):
        """Returns description of Model"""
        return f"Model: {self.layers}"
    
    def _set_cache(self):
        """Creates a cache for optimization."""
        self.cache_w = []
        self.cache_b = []
        for l in self.layers:
            self.cache_w.append(np.zeros_like(l.w))
            self.cache_b.append(np.zeros_like(l.b))
    
    def _set_buffer(self):
        """Creates a buffer for gradient additions."""
        self._gb_w = []
        self._gb_b = []
        for l in self.layers:
            self._gb_w.append(np.zeros_like(l.w))
            self._gb_b.append(np.zeros_like(l.b))
    
    def forward(self, x):
        """Forward pass through all layers.
        Returns a list of all passes/inputs."""
        passes = []
        passes.append(x)
        for layer in self.layers:
            x = layer(x)
            passes.append(x)
        return passes
    
    def backward(self, passes, grad):
        """Backwards pass through all the layers.
        Reverses passes and runs trough grad calculation with transposed pass.
        Returns list of update gradients.
        """
        passes.reverse()
        
        prev_grad = grad
        
        dw = []
        db = []
        
        for ind, layer in enumerate(reversed(self.layers)):
            w_grad, b_grad, prev_grad = layer._backwards(prev_grad, passes[ind].T)
            
            dw.insert(0, w_grad)
            db.insert(0, b_grad)
        
        return dw, db

    def _loss(self, label, pred, d=False):
        """Calculating loss.
        MSE (Mean Squared Error):
        - MSE = mean((label - prediction) ^ 2)
        - dMSE/dlabel = 2 * (label - pred) / n
        MAE (Mean Absolute Error):
        - MAE = sum(|label - prediction|)
        - dMAE/dlabel = -1 when lable - pred > 0 and 1 when label - pred < 0
        Normal:
        - Loss = label - pred
        - dLoss/dlabel = 1"""
        
        if self.lt == 'mse':
            if d:
                return -2 * (label - pred) / len(label)
            return np.mean((label - pred) ** 2)
        
        if self.lt == 'mae':
            if d:
                return -1/len(label) * np.sign(label - pred)
            return np.mean(np.abs(label - pred))
        
        if d:
            return 1
        return label - pred
    
    def _reward(self, actions, reward):
        """Calculating reward.
        Discounts rewards, averages them, and divides by standard deviation.
        Finds gradients in likeness loss by getting negative."""
        
        drw = self._discount_rewards(reward)
        drw -= np.mean(drw)
        drw /= (np.std(drw) + 1e-8)
        
        return actions * -drw
    
    def _discount_rewards(self, rewards):
        """Discounting rewards,
        Gives more reward depending on recency.
        Discounts by discount value set by config.
        """
        
        discounted_reward = np.zeros_like(rewards, dtype=np.float64)
        total_rewards = 0
        for t in reversed(range(0, len(rewards))):
            total_rewards = total_rewards * self.discount + rewards[t]
            discounted_reward[t] = total_rewards
        return discounted_reward
    
    def _optimize(self, dw, db):
        """Optimizes gradient descent.
        Returns weight updates and bias updates.
        RMSPROP: root mean squared propagation
        SGD: scholastic gradient descent"""
        
        if self.optm == 'rmsprop':
            dw_u = []
            db_u = []
            for i in range(len(self.layers)):
                g_w = dw[i]
                self.cache_w[i] = self.cache_w[i] * self.beta[0] + (1 - self.beta[0]) * g_w ** 2
                dw_u.append(self.lr * g_w / np.sqrt(self.cache_w[i] + 1e-8))
                
                g_b = db[i]
                self.cache_b[i] = self.cache_b[i] * self.beta[0] + (1 - self.beta[0]) * g_b ** 2
                db_u.append(self.lr * g_b / np.sqrt(self.cache_b[i] + 1e-8))
            
            return dw_u, db_u
        
        for i in range(len(dw)):
            dw[i] = dw[i] * self.lr
            db[i] = db[i] * self.lr
        return dw, db

    def _fix_hid(self, hid):
        """Reorganizes the hidden layers from epoch list to layer list."""
        
        vhidden = []
        for layer in range(len(hid[0])):
            one_layer = []
            for i in range(len(hid)):
                one_layer.append(hid[i][layer])
            vhidden.append(np.array(one_layer))
        return vhidden
    
    def train(self, config):
        """Trains the algorithm.
        Config: Dict
        - 'epochs': amount of training episodes
        - 'batch': batch size (increases/decreases volitility)
        - 'env': environment to train the algorithm"""
        
        self._batch_size = config.get('batch', 10)
        _epoch = config.get('epochs', 1000)
        env = config.get('env', None)
        
        self._set_buffer()
        self._set_cache()
        
        o_list = []
        weight_history = []
        
        for epoch in range(_epoch):
            o = None
            if self.v == 'loss':
                o = self._epoch_loss(epoch, env)
            if self.v == 'reward':
                o = self._epoch_reward(epoch, env)
            utility.progress_bar(epoch + 1, _epoch, suffix=f'{self.v.upper()}: {round(o, 5)}', length=50)
            o_list.append(o)
            
            epoch_weights = []
            for layer in self.layers:
                w, b = layer.parameters()
                epoch_weights.append(w.flatten())
            weight_history.append(epoch_weights)
        
        if self.v == 'loss':
            x_train, y_train = env.get_train()
            
            passes = self.forward(x_train)
            y_pred = passes.pop(-1)
        
            loss = self._loss(y_train, y_pred)
            
            accuracy = np.mean([(yi > 0) == (scorei > 0) for yi, scorei in zip(y_train, y_pred)])
            
            print(f"Test: Loss ({loss}), Accuracy ({accuracy * 100}%)")
                   
        if self.v == 'reward':
            state = env.reset()
            old_state = 0
            reward_sum = 0
            done = False
            
            while not done:
                calc_state = state - old_state
                
                if self.cont:
                    old_state = state
                
                passes = self.forward(calc_state)
                act, act_grad = self.sample_action(passes.pop(-1))
                
                state, reward, done, _ = env.step(act)
                                
                reward_sum += reward
            
            print(f"Test: Reward Sum ({reward_sum})")
        
        return o_list, weight_history

    def visualize(self, o_list, weight_history):
        plt.figure()
        plt.plot(o_list)
        plt.title('Plot of o_list')
        plt.xlabel('Index')
        plt.ylabel('Values')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        x, y, z = [], [], []

        for epoch_history in weight_history:
            x.append([weights[0] for weights in epoch_history])
            y.append([weights[1] for weights in epoch_history])
            z.append([weights[2] for weights in epoch_history])
        
        col = ['red', 'blue', 'green', 'yellow', 'orange', 'cyan']
        for i in range(len(x[0])):
            x_i = [x[ind][i] for ind in range(len(x))]
            y_i = [y[ind][i] for ind in range(len(y))]
            z_i = [z[ind][i] for ind in range(len(z))]
            ax.scatter(x_i, y_i, z_i, c=col[i], marker='.')
            ax.scatter([x_i[0]], [y_i[0]], [z_i[0]], c=col[i], marker='o', s=75, label=f'Start Pt of Layer {i}')
            ax.scatter([x_i[-1]], [y_i[-1]], [z_i[-1]], c=col[i], marker='^', s=75, label=f'End Pt of Layer {i}')
        
        ax.set_title('3D Scatter Plot of Weight History')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        plt.show()
        
    def _epoch_reward(self, epoch, env):
        """A single run through an epoch (reward based)"""
        
        shidden, sgrads, srewards = [], [], []
        state = env.reset()
        old_state = 0
        reward_sum = 0
        done = False
        
        while not done:
            calc_state = state - old_state
            
            if self.cont:
                old_state = state
            
            passes = self.forward(calc_state)
            act, act_grad = self.sample_action(passes.pop(-1))
            
            shidden.append(passes)
            sgrads.append(act_grad)
            
            state, reward, done, _ = env.step(act)
            
            srewards.append(reward)
            
            reward_sum += reward
        
        vgrads = np.array(sgrads)
        vrewards = np.vstack(srewards)
        
        vgrads = self._reward(vgrads, vrewards)
        
        self._update_weights(epoch, self._fix_hid(shidden), vgrads)
        
        return reward_sum
    
    def _epoch_loss(self, epoch, env):
        """A single run through an epoch (loss based)"""
        
        x_train, y_train = env.get_train()
        
        passes = self.forward(x_train)
        y_pred = passes.pop(-1)
        
        loss = self._loss(y_train, y_pred)
        grad = self._loss(y_train, y_pred, True)
        
        self._update_weights(epoch, passes, grad)
        
        return loss

    def _update_weights(self, epoch, passes, grad):
        """Updates buffer based on passes and grad and updates weights if batch is over."""
        g_w, g_b = self.backward(passes, grad)
        for i in range(len(self._gb_w)):
            self._gb_w[i] += g_w[i]
            self._gb_b[i] += g_b[i]
        
        if epoch % self._batch_size == 0:
            dw, db = self._optimize(self._gb_w, self._gb_b)
            for i in range(len(self.layers)):
                self.layers[i].w -= dw[i]
                self.layers[i].b -= db[i]
            self._set_buffer()
    
    def sample_action(self, x):
        """Return an action and action_grad."""
        e_x = np.exp(x - np.max(x))
        e_x = e_x / e_x.sum(axis=0)
        if random.random() < 0.05:
            act = random.randint(0, len(x) - 1)
        else:
            act = np.argmax(e_x)
        aoh = np.zeros_like(x)
        aoh[act] = 1
        a_g = aoh - e_x
        
        return act, a_g

