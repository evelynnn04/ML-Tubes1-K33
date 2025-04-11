from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns 
import math

'''
Note:
- if y_true.ndim == 1: y_true = y_true.reshape(-1, 1) -> Kalo array 1D ubah jadi array 2D
- if y_true.shape != y_pred.shape: y_true = np.eye(y_pred.shape[1])[y_true.flatten()] -> handle kalo y_true contain class label bukan one hot 
'''

def mse(y_true, y_pred):
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_true.shape != y_pred.shape:
        y_true = np.eye(y_pred.shape[1])[y_true.flatten()]
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_true.shape != y_pred.shape:
        y_true = np.eye(y_pred.shape[1])[y_true.flatten()]
    return 2 * (y_pred - y_true) / y_true.shape[0] 

def bce(y_true, y_pred):
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_true.shape != y_pred.shape:
        y_true = np.eye(y_pred.shape[1])[y_true.flatten()]
    
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def bce_derivative(y_true, y_pred):
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_true.shape != y_pred.shape:
        y_true = np.eye(y_pred.shape[1])[y_true.flatten()]
    
    epsilon = 1e-7  
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / (y_pred * (1 - y_pred) + epsilon)

def cce(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
        y_true = np.eye(y_pred.shape[1])[y_true.flatten()]
    
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def cce_derivative(y_true, y_pred):
    if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
        y_true = np.eye(y_pred.shape[1])[y_true.flatten()]
    
    return y_pred - y_true

class Layer:
    def __init__(self, n_neurons, init='zero', activation='linear', init_params=None, weights=None, biases=None, use_rmsnorm=False, rmsnorm_eps=1e-8):
        """
        Initialize a neural network layer
        
        Parameters:
        -----------
        n_neurons : int
            Number of neurons in the layer
        init : str, optional (default='zero')
            Initialization method. Options:
            - 'zero': Zero initialization
            - 'uniform': Uniform random distribution
            - 'normal': Normal (Gaussian) random distribution
        activation : str, optional (default='linear')
            Activation function to use
        init_params : dict, optional
            Additional parameters for initialization:
            - For 'uniform': 
                * 'lower': lower bound (default: -1)
                * 'upper': upper bound (default: 1)
                * 'seed': random seed (optional)
            - For 'normal':
                * 'mean': mean of distribution (default: 0)
                * 'variance': variance of distribution (default: 1)
                * 'seed': random seed (optional)
        use_rmsnorm : bool, optional (default=False)
            Whether to use RMSNorm normalization for this layer
        rmsnorm_eps : float, optional (default=1e-8)
            Small constant added to denominator in RMSNorm for numerical stability
        """
        self.n_neurons = n_neurons
        self.init = init
        self.activation = activation
        self.init_params = init_params or {}
        self.use_rmsnorm = use_rmsnorm
        self.rmsnorm_eps = rmsnorm_eps
        self.rmsnorm_scale = None
        
        if self.init == 'uniform':
            self.init_params.setdefault('lower', -1)
            self.init_params.setdefault('upper', 1)
        elif self.init == 'normal':
            self.init_params.setdefault('mean', 0)
            self.init_params.setdefault('variance', 1)
        
        self.weights = weights
        self.biases = biases
    
    def initialize(self, input_dim):
        if 'seed' in self.init_params:
            np.random.seed(self.init_params['seed'])

        if self.init == 'zero':
            self.weights = np.zeros((input_dim, self.n_neurons))
            self.biases = np.zeros((1, self.n_neurons))
        
        elif self.init == 'uniform':
            lower = self.init_params['lower']
            upper = self.init_params['upper']
            self.weights = np.random.uniform(low=lower, high=upper, size=(input_dim, self.n_neurons))
            self.biases = np.random.uniform(low=lower, high=upper, size=(1, self.n_neurons))
        
        elif self.init == 'normal':
            mean = self.init_params['mean']
            variance = self.init_params['variance']
            self.weights = np.random.normal(loc=mean, scale=np.sqrt(variance), size=(input_dim, self.n_neurons))
            self.biases = np.random.normal(low=lower, high=upper, size=(1, self.n_neurons))
        
        elif self.init == 'xavier_uniform':
            limit = np.sqrt(6 / (input_dim + self.n_neurons))
            self.weights = np.random.uniform(-limit, limit, (input_dim, self.n_neurons))
            self.biases = np.random.uniform(-limit, limit, size=(1, self.n_neurons))
        
        elif self.init == 'xavier_normal':
            std = np.sqrt(2 / (input_dim + self.n_neurons))
            self.weights = np.random.normal(0, std, (input_dim, self.n_neurons))
            self.biases = np.random.normal(0, std, size=(1, self.n_neurons))
        
        elif self.init == 'he_normal':
            std = np.sqrt(2 / input_dim)
            self.weights = np.random.normal(0, std, (input_dim, self.n_neurons))
            self.biases = np.random.normal(0, std, (1, self.n_neurons))
        
        elif self.init == 'he_uniform':
            limit = np.sqrt(6 / input_dim)
            self.weights = np.random.uniform(-limit, limit, (input_dim, self.n_neurons))
            self.biases = np.random.uniform(-limit, limit, (1, self.n_neurons))

        else:
            raise ValueError(
                f"Unknown initialization type: {self.init}\n"
                "Available types: zero, uniform, normal, xavier_uniform, xavier_normal, he_normal, he_uniform"
            )
        
        if self.use_rmsnorm:
            self.rmsnorm_scale = np.ones((1, self.n_neurons))
        
        return self
        
    def activate(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'leaky_relu':
            return np.where(x > 0, x, 0.01 * x)
        elif self.activation == 'elu':
            return np.where(x > 0, x, 0.01 * (np.exp(x) - 1))
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        else:
            raise ValueError(
                f"Unknown activation function: {self.activation}\n"
                "/n Activation function available: linear, relu,sigmoid, tanh, softmax, elu, leaky_relu"
            )
    
    def activation_derivative(self, x):
        if self.activation == 'linear':
            return np.ones_like(x)
        elif self.activation == 'relu':
            return (x > 0).astype(float)
        elif self.activation == 'leaky_relu':
            return np.where(x > 0, 1, 0.01)
        elif self.activation == 'elu':
            alpha = 0.01
            return np.where(x > 0, 1, alpha * np.exp(x))
        elif self.activation == 'sigmoid':
            s = self.activate(x)
            return s * (1 - s)
        elif self.activation == 'tanh':
            t = np.tanh(x)
            return 1 - t**2
        elif self.activation == 'softmax':
            return 1
        else:
            raise ValueError(
                f"Unknown activation function: {self.activation}/n"
                "Activation function available: linear, relu,sigmoid, tanh, softmax, elu, leaky_relu"
            )

class FFNN:
    def __init__(self, loss='mse', batch_size=32, learning_rate=0.01, epochs=100, verbose=1, l1_lambda=0, l2_lambda=0):
        self.layers = []
        self.learning_rate = learning_rate
        self.loss = loss
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.train_losses = []
        self.val_losses = []
        self.weights = []
        self.biases = []
        self.gradient_weights = []
        self.gradient_biases = []
        self.gradient_rmsnorm_scale = []
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        
        if loss == 'mse':
            self.loss_func = mse
            self.loss_derivative = mse_derivative
        elif loss == 'bce':
            self.loss_func = bce
            self.loss_derivative = bce_derivative
        elif loss == 'cce':
            self.loss_func = cce
            self.loss_derivative = cce_derivative
        else:
            raise ValueError(
                f"Unknown loss function: {loss}\n"
                "Loss function available: mse, bce, cce"
            )
    
    def build_layers(self, *layer_args):
        self.layers = list(layer_args)
    
    def _initialize_network(self, input_dim):
        prev_dim = input_dim
        for layer in self.layers:
            layer.initialize(prev_dim)
            prev_dim = layer.n_neurons
    
    def _compute_regularization_loss(self):
        """Calculate the regularization component of the loss"""
        l1_reg = 0
        l2_reg = 0
        
        if self.l1_lambda > 0:
            for layer in self.layers:
                l1_reg += np.sum(np.abs(layer.weights))
            l1_reg *= self.l1_lambda
            
        if self.l2_lambda > 0:
            for layer in self.layers:
                l2_reg += np.sum(np.square(layer.weights))
            l2_reg *= 0.5 * self.l2_lambda  # 0.5 is conventional for L2
            
        return l1_reg + l2_reg
    
    def _compute_total_loss(self, y, y_pred):
        """Compute the total loss including regularization"""
        base_loss = self.loss_func(y, y_pred)
        reg_loss = self._compute_regularization_loss()
        return base_loss + reg_loss
    
    def forward(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        activations = [X]
        zs = []
        normalized_zs = []
        
        for layer in self.layers:
            z = activations[-1] @ layer.weights + layer.biases # @ is dot product
            
            if layer.use_rmsnorm:
                zs.append(z)
                
                rms = np.sqrt(np.mean(z**2, axis=1, keepdims=True) + layer.rmsnorm_eps)
                
                z_norm = z / rms
                z = z_norm * layer.rmsnorm_scale
                
                normalized_zs.append(z_norm)
            else:
                zs.append(z)
                normalized_zs.append(None)
            
            a = layer.activate(z)
            activations.append(a)
            
        return zs, activations, normalized_zs
    
    def backward(self, X, y, zs, activations, normalized_zs):
        m = X.shape[0]
        y_pred = activations[-1]
        
        self.weights = []
        self.biases = []
        self.gradient_weights = []
        self.gradient_biases = []
        self.gradient_rmsnorm_scale = []
        
        delta = self.loss_derivative(y, y_pred)
        
        for i in reversed(range(len(self.layers))):
            z = zs[i]
            a_prev = activations[i]
            z_norm = normalized_zs[i]
            
            if self.layers[i].use_rmsnorm and z_norm is not None:
                grad_scale = np.sum(delta * z_norm, axis=0, keepdims=True) / m
                self.gradient_rmsnorm_scale.append(grad_scale)
                
                rms = np.sqrt(np.mean(z**2, axis=1, keepdims=True) + self.layers[i].rmsnorm_eps)
                
                dim = z.shape[1]
                delta_z = delta * self.layers[i].rmsnorm_scale 
                grad_rms_1 = delta_z / rms
                
                z_squared_sum = np.sum(z**2, axis=1, keepdims=True)
                grad_rms_2 = -delta_z * z * (1.0 / (dim * rms**3 * z_squared_sum + self.layers[i].rmsnorm_eps))
                
                delta = grad_rms_1 + grad_rms_2
                
                self.layers[i].rmsnorm_scale -= self.learning_rate * grad_scale
            else:
                self.gradient_rmsnorm_scale.append(None)
            
            grad_w = (a_prev.T @ delta) / m
            grad_b = np.sum(delta, axis=0, keepdims=True) / m
            
            if self.l1_lambda > 0:
                l1_grad = self.l1_lambda * np.sign(self.layers[i].weights)
                grad_w += l1_grad
            
            if self.l2_lambda > 0:
                l2_grad = self.l2_lambda * self.layers[i].weights
                grad_w += l2_grad

            self.weights.append(self.layers[i].weights.copy())
            self.biases.append(self.layers[i].biases.copy())
            
            self.gradient_weights.append(grad_w)
            self.gradient_biases.append(grad_b)
            
            self.layers[i].weights -= self.learning_rate * grad_w
            self.layers[i].biases -= self.learning_rate * grad_b
            
            if i > 0:
                delta = (delta @ self.layers[i].weights.T) * self.layers[i - 1].activation_derivative(zs[i - 1])
        
        self.weights.reverse()
        self.biases.reverse()
        self.gradient_weights.reverse()
        self.gradient_biases.reverse()
        self.gradient_rmsnorm_scale.reverse()
    
    def fit(self, X, y, X_val=None, y_val=None):
        self._initialize_network(X.shape[1])
        
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            X_train, y_train = X, y
        
        if self.verbose == 0:
            for epoch in range(self.epochs):
                indices = np.arange(X.shape[0])
                np.random.shuffle(indices)
                
                for start in range(0, X.shape[0], self.batch_size):
                    end = start + self.batch_size
                    batch_indices = indices[start:end]
                    X_batch = X[batch_indices]
                    y_batch = y[batch_indices]
                    
                    zs, activations, normalized_zs = self.forward(X_batch)
                    self.backward(X_batch, y_batch, zs, activations, normalized_zs)

                zs_train, activations_train, _ = self.forward(X)
                y_train_pred = activations_train[-1]
                train_loss = self._compute_total_loss(y, y_train_pred)
                
                zs_val, activations_val, _ = self.forward(X_val)
                y_val_pred = activations_val[-1]
                val_loss = self._compute_total_loss(y_val, y_val_pred)

                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
        
        elif self.verbose == 1:
            for epoch in range(self.epochs):
                epoch_progress = tqdm(total=X.shape[0], desc=f"Epoch {epoch+1}/{self.epochs}", unit='sample')
                
                indices = np.arange(X.shape[0])
                np.random.shuffle(indices)
                
                for start in range(0, X.shape[0], self.batch_size):
                    end = start + self.batch_size
                    batch_indices = indices[start:end]
                    X_batch = X[batch_indices]
                    y_batch = y[batch_indices]
                    
                    zs, activations, normalized_zs = self.forward(X_batch)
                    self.backward(X_batch, y_batch, zs, activations, normalized_zs)

                    epoch_progress.update(len(X_batch))
                
                epoch_progress.close()
                
                zs_train, activations_train, _ = self.forward(X)
                y_train_pred = activations_train[-1]
                train_loss = self._compute_total_loss(y, y_train_pred)
                
                zs_val, activations_val, _ = self.forward(X_val)
                y_val_pred = activations_val[-1]
                val_loss = self._compute_total_loss(y_val, y_val_pred)

                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                
                print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        else:
            raise ValueError(
                f"Invalid verbose value: {self.verbose}\n"
                "Verbose options: 0 (no output), 1 (progress bar)"
            )
    
    def predict(self, X):
        _, activations, _ = self.forward(X)
        if self.loss == 'cce':  
            return np.argmax(activations[-1], axis=1)
        return activations[-1]
    
    def save(self, filename):
        model_state = {
            'layers': self.layers,
            'learning_rate': self.learning_rate,
            'loss': self.loss,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'verbose': self.verbose,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'weights': self.weights,
            'biases': self.biases,
            'gradient_weights': self.gradient_weights,
            'gradient_biases': self.gradient_biases,
            'gradient_rmsnorm_scale': getattr(self, 'gradient_rmsnorm_scale', [None] * len(self.layers)),
            'l1_lambda': self.l1_lambda,
            'l2_lambda': self.l2_lambda
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_state, f)
        
        print(f"Model saved to {filename}")

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            model_state = pickle.load(f)
        
        model = cls(
            loss=model_state['loss'],
            batch_size=model_state['batch_size'],
            learning_rate=model_state['learning_rate'],
            epochs=model_state['epochs'],
            verbose=model_state['verbose'],
            l1_lambda=model_state.get('l1_lambda', 0),  
            l2_lambda=model_state.get('l2_lambda', 0)  
        )
        
        model.layers = model_state['layers']
        model.train_losses = model_state['train_losses']
        model.val_losses = model_state['val_losses']
        model.weights = model_state['weights']
        model.biases = model_state['biases']
        model.gradient_weights = model_state['gradient_weights']
        model.gradient_biases = model_state['gradient_biases']
        
        print(f"Model loaded from {filename}")
        return model

    # TODO: imporove this
    def visualize_architecture(self, output_file='neural_network.png', figsize=(12, 8)):
        G = nx.DiGraph()
        layers = []
        
        if len(self.layers) > 0:
            input_dim = self.layers[0].weights.shape[0]
            input_layer = [f'Input {i}' for i in range(input_dim)]
            layers.append(input_layer)
        
        for layer_idx, layer in enumerate(self.layers):
            layer_nodes = [f'Layer {layer_idx} Neuron {i}' for i in range(layer.n_neurons)]
            layers.append(layer_nodes)
        
        plt.figure(figsize=figsize)
        
        pos = {}
        max_nodes_in_layer = max(len(layer) for layer in layers)
        
        for layer_idx, layer_nodes in enumerate(layers):
            y_spacing = 1 / (len(layer_nodes) + 1)
            for node_idx, node in enumerate(layer_nodes):
                y = 1 - (node_idx + 1) * y_spacing
                x = layer_idx / (len(layers) - 1)
                pos[node] = (x, y)
                G.add_node(node)
        
        # Add edges between layers
        for layer_idx in range(len(layers) - 1):
            for prev_node in layers[layer_idx]:
                for curr_node in layers[layer_idx + 1]:
                    prev_idx = layers[layer_idx].index(prev_node)
                    curr_idx = layers[layer_idx + 1].index(curr_node)

                    if layer_idx < len(self.layers):
                        weight = self.layers[layer_idx].weights[prev_idx, curr_idx]
                        bias = self.layers[layer_idx].biases[0, curr_idx]
                        G.add_edge(prev_node, curr_node, weight=weight, bias=bias)
        
        plt.title("Neural Network Architecture")
        
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500, alpha=0.8)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")
        
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        min_weight, max_weight = min(weights), max(weights)
        normalized_weights = [(w - min_weight) / (max_weight - min_weight) for w in weights]
        
        nx.draw_networkx_edges(
            G, pos, 
            edge_color=[plt.cm.RdYlGn(w) for w in normalized_weights],
            width=[2 + 5 * w for w in normalized_weights],
            alpha=0.7,
            arrows=True
        )
        
        edge_labels = {(u, v): f'w: {G[u][v]["weight"]:.2f}\nb: {G[u][v]["bias"]:.2f}' 
                       for (u, v) in G.edges()}
        nx.draw_networkx_edge_labels(
            G, pos, 
            edge_labels=edge_labels, 
            font_color='red', 
            font_size=6
        )
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_weight_distribution(self, layers_to_plot=None, figsize=(6, 4)):

        if layers_to_plot is None:
            layers_to_plot = list(range(len(self.weights)))
        
        layers_to_plot = [layer for layer in layers_to_plot if 0 <= layer < len(self.weights)]
        
        num_layers = len(layers_to_plot)
        num_cols = min(num_layers, 3)
        num_rows = math.ceil(num_layers / num_cols)
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(figsize[0] * num_cols, figsize[1] * num_rows))

        if num_rows > 1:
            axes = axes.flatten()
        elif num_layers == 1:
            axes = [axes]
        
        for i in range(num_layers, len(axes)):
            fig.delaxes(axes[i])
        
        for i, layer_idx in enumerate(layers_to_plot):
            weights = self.weights[layer_idx]
            flat_weights = weights.flatten()
            
            sns.histplot(flat_weights, kde=True, ax=axes[i])
            
            axes[i].set_title(f'Layer {layer_idx} Weight Distribution')
            axes[i].set_xlabel('Weight Values')
            axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        return fig
    
    def plot_biases_distribution(self, layers_to_plot=None, figsize=(6, 4)):

        if layers_to_plot is None:
            layers_to_plot = list(range(len(self.biases)))
        
        layers_to_plot = [layer for layer in layers_to_plot if 0 <= layer < len(self.biases)]
        
        num_layers = len(layers_to_plot)
        num_cols = min(num_layers, 3)
        num_rows = math.ceil(num_layers / num_cols)
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(figsize[0] * num_cols, figsize[1] * num_rows))

        if num_rows > 1:
            axes = axes.flatten()
        elif num_layers == 1:
            axes = [axes]
        
        for i in range(num_layers, len(axes)):
            fig.delaxes(axes[i])
        
        for i, layer_idx in enumerate(layers_to_plot):
            biases = self.biases[layer_idx]
            flat_biases = biases.flatten()
            
            sns.histplot(flat_biases, kde=True, ax=axes[i])
            
            axes[i].set_title(f'Layer {layer_idx} Biases')
            axes[i].set_xlabel('Bias Values')
            axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        return fig
    
    def plot_gradient_weight_distribution(self, layers_to_plot=None, figsize=(6, 4)):

        if layers_to_plot is None:
            layers_to_plot = list(range(len(self.gradient_weights)))
        
        layers_to_plot = [layer for layer in layers_to_plot if 0 <= layer < len(self.gradient_weights)]
        
        num_layers = len(layers_to_plot)
        num_cols = min(num_layers, 3)
        num_rows = math.ceil(num_layers / num_cols)
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(figsize[0] * num_cols, figsize[1] * num_rows))

        if num_rows > 1:
            axes = axes.flatten()
        elif num_layers == 1:
            axes = [axes]
        
        for i in range(num_layers, len(axes)):
            fig.delaxes(axes[i])
        
        for i, layer_idx in enumerate(layers_to_plot):
            gradient_weights = self.gradient_weights[layer_idx]
            flat_gradient_weights = gradient_weights.flatten()
            
            sns.histplot(flat_gradient_weights, kde=True, ax=axes[i])
            
            axes[i].set_title(f'Layer {layer_idx} Gradient Weight')
            axes[i].set_xlabel('Gradient Weight Values')
            axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        return fig
    
    def plot_gradient_biases_distribution(self, layers_to_plot=None, figsize=(6, 4)):

        if layers_to_plot is None:
            layers_to_plot = list(range(len(self.gradient_biases)))
        
        layers_to_plot = [layer for layer in layers_to_plot if 0 <= layer < len(self.gradient_biases)]
        
        num_layers = len(layers_to_plot)
        num_cols = min(num_layers, 3)
        num_rows = math.ceil(num_layers / num_cols)
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(figsize[0] * num_cols, figsize[1] * num_rows))

        if num_rows > 1:
            axes = axes.flatten()
        elif num_layers == 1:
            axes = [axes]
        
        for i in range(num_layers, len(axes)):
            fig.delaxes(axes[i])
        
        for i, layer_idx in enumerate(layers_to_plot):
            gradient_biases = self.gradient_biases[layer_idx]
            flat_gradient_biases = gradient_biases.flatten()
            
            sns.histplot(flat_gradient_biases, kde=True, ax=axes[i])
            
            axes[i].set_title(f'Layer {layer_idx} Gradient Biases')
            axes[i].set_xlabel('Bias Values')
            axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        return fig


    def plot_training_loss(self):
        if not self.train_losses or not self.val_losses:
            print("Loss history is empty. Train the model first.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title("Training and Validation Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
