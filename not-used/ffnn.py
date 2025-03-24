import numpy as np
import matplotlib.pyplot as plt
import pickle
import networkx as nx
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

class FFNN:
    def __init__(self, layer_sizes, activation_functions, loss_function, weight_init_method='uniform', 
                 lower_bound=-0.1, upper_bound=0.1, seed=42):
        """
        Initialize a Feedforward Neural Network.
        
        Parameters:
        -----------
        layer_sizes : list
            Number of neurons in each layer (including input and output layers)
        activation_functions : list
            Activation function for each layer (including input layer)
        loss_function : str
            Loss function to use ('binary_cross_entropy')
        weight_init_method : str
            Method to initialize weights ('uniform')
        lower_bound : float
            Lower bound for weight initialization (if using uniform)
        upper_bound : float
            Upper bound for weight initialization (if using uniform)
        seed : int
            Random seed for reproducibility
        """
        self.layer_sizes = layer_sizes
        
        # Make sure activation_functions has the correct length
        if len(activation_functions) != len(layer_sizes):
            # If not enough activation functions provided, add 'linear' for input layer
            if len(activation_functions) == len(layer_sizes) - 1:
                activation_functions = ['linear'] + activation_functions
            else:
                raise ValueError(f"Expected {len(layer_sizes)} activation functions, got {len(activation_functions)}")
        
        self.activation_functions = activation_functions
        self.loss_function = loss_function
        self.weight_init_method = weight_init_method
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.seed = seed
        
        # Initialize weights and biases
        np.random.seed(seed)
        self.weights = []
        self.biases = []
        self.weight_gradients = []
        self.bias_gradients = []
        
        # Initialize weights between layers
        for i in range(len(layer_sizes) - 1):
            if weight_init_method == 'uniform':
                w = np.random.uniform(lower_bound, upper_bound, (layer_sizes[i], layer_sizes[i+1]))
                b = np.random.uniform(lower_bound, upper_bound, (1, layer_sizes[i+1]))
            else:
                raise ValueError(f"Initialization method {weight_init_method} not supported.")
            
            self.weights.append(w)
            self.biases.append(b)
            self.weight_gradients.append(np.zeros_like(w))
            self.bias_gradients.append(np.zeros_like(b))
        
        # For storing activations and pre-activations during forward pass
        self.activations = []
        self.pre_activations = []
        
        # Training history
        self.history = {'train_loss': [], 'val_loss': []}
    
    def _linear(self, x):
        """Linear activation function (identity)"""
        return x
    
    def _linear_derivative(self, x):
        """Derivative of linear function"""
        return np.ones_like(x)
    
    def _relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        """Derivative of ReLU function"""
        return np.where(x > 0, 1, 0)
    
    def _tanh(self, x):
        """Hyperbolic tangent activation function"""
        return np.tanh(x)
    
    def _tanh_derivative(self, x):
        """Derivative of hyperbolic tangent function"""
        return 1 - np.tanh(x) ** 2
    
    def _sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
    
    def _sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        s = self._sigmoid(x)
        return s * (1 - s)
    
    def _binary_cross_entropy(self, y_true, y_pred):
        """Binary cross-entropy loss function"""
        # Clip predictions to avoid log(0) or log(1)
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        # Calculate BCE loss
        if y_true.shape[1] == 1:  # Binary classification
            loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:  # Multi-class classification
            loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return loss
    
    def _apply_activation(self, x, activation_name):
        """Apply activation function to input"""
        if activation_name == 'linear':
            return self._linear(x)
        elif activation_name == 'relu':
            return self._relu(x)
        elif activation_name == 'tanh':
            return self._tanh(x)
        elif activation_name == 'sigmoid':
            return self._sigmoid(x)
        else:
            raise ValueError(f"Activation function {activation_name} not supported.")
    
    def _apply_activation_derivative(self, x, activation_name):
        """Apply derivative of activation function to input"""
        if activation_name == 'linear':
            return self._linear_derivative(x)
        elif activation_name == 'relu':
            return self._relu_derivative(x)
        elif activation_name == 'tanh':
            return self._tanh_derivative(x)
        elif activation_name == 'sigmoid':
            return self._sigmoid_derivative(x)
        else:
            raise ValueError(f"Activation function {activation_name} not supported.")
    
    def forward(self, X):
        """
        Forward propagation through the network.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data, shape (batch_size, input_size)
        
        Returns:
        --------
        numpy.ndarray
            Output predictions, shape (batch_size, output_size)
        """
        # Reset activations and pre-activations
        self.activations = [X]  # Initial activation is just the input
        self.pre_activations = []
        
        # Forward pass through each layer
        for i in range(len(self.weights)):
            # Compute pre-activation: z = a^(l-1) * W^(l) + b^(l)
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.pre_activations.append(z)
            
            # Apply activation function: a^(l) = activation(z^(l))
            if i == len(self.weights) - 1 and self.loss_function == 'binary_cross_entropy':
                # For the output layer with binary cross-entropy, use sigmoid
                a = self._sigmoid(z)
            else:
                # For other layers, use the specified activation function
                a = self._apply_activation(z, self.activation_functions[i+1])
            
            self.activations.append(a)
        
        return self.activations[-1]
    
    def backward(self, y_true):
        """
        Backward propagation to compute gradients.
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            True labels, shape (batch_size, output_size)
        """
        batch_size = y_true.shape[0]
        
        # Initialize delta for output layer
        if self.loss_function == 'binary_cross_entropy':
            # For binary cross-entropy with sigmoid output
            delta = (self.activations[-1] - y_true) / batch_size
        else:
            raise ValueError(f"Loss function {self.loss_function} not supported.")
        
        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Calculate gradients for current layer
            self.weight_gradients[i] = np.dot(self.activations[i].T, delta)
            self.bias_gradients[i] = np.sum(delta, axis=0, keepdims=True)
            
            # Calculate delta for previous layer (if not at input layer)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self._apply_activation_derivative(
                    self.pre_activations[i-1], self.activation_functions[i]
                )
    
    def update_weights(self, learning_rate):
        """
        Update weights and biases using computed gradients.
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate for gradient descent
        """
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * self.weight_gradients[i]
            self.biases[i] -= learning_rate * self.bias_gradients[i]
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, batch_size=32, learning_rate=0.01, epochs=10, verbose=1):
        """
        Train the neural network.
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training data
        y_train : numpy.ndarray
            Training labels
        X_val : numpy.ndarray
            Validation data
        y_val : numpy.ndarray
            Validation labels
        batch_size : int
            Batch size for training
        learning_rate : float
            Learning rate for gradient descent
        epochs : int
            Number of training epochs
        verbose : int
            Verbosity level (0: no output, 1: progress bar)
        
        Returns:
        --------
        dict
            Training history
        """
        num_samples = X_train.shape[0]
        
        # Ensure y_train is 2D
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
        
        # Ensure y_val is 2D if provided
        if X_val is not None and y_val is not None and len(y_val.shape) == 1:
            y_val = y_val.reshape(-1, 1)
        
        # Reset history
        self.history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Initialize epoch loss
            epoch_loss = 0
            
            # Create progress bar if verbose
            if verbose == 1:
                progress_bar = tqdm(range(0, num_samples, batch_size), desc=f"Epoch {epoch+1}/{epochs}")
            else:
                progress_bar = range(0, num_samples, batch_size)
            
            # Train on mini-batches
            for batch_start in progress_bar:
                batch_end = min(batch_start + batch_size, num_samples)
                X_batch = X_shuffled[batch_start:batch_end]
                y_batch = y_shuffled[batch_start:batch_end]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Calculate loss
                batch_loss = self._binary_cross_entropy(y_batch, y_pred)
                epoch_loss += batch_loss * (batch_end - batch_start) / num_samples
                
                # Backward pass
                self.backward(y_batch)
                
                # Update weights
                self.update_weights(learning_rate)
                
                # Update progress bar if verbose
                if verbose == 1:
                    progress_bar.set_postfix({"train_loss": f"{epoch_loss:.4f}"})
            
            # Append training loss to history
            self.history['train_loss'].append(epoch_loss)
            
            # Calculate validation loss if validation data is provided
            if X_val is not None and y_val is not None:
                val_pred = self.forward(X_val)
                val_loss = self._binary_cross_entropy(y_val, val_pred)
                self.history['val_loss'].append(val_loss)
                
                if verbose == 1:
                    print(f"Epoch {epoch+1}/{epochs} - train_loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f}")
            else:
                if verbose == 1:
                    print(f"Epoch {epoch+1}/{epochs} - train_loss: {epoch_loss:.4f}")
        
        return self.history
    
    def predict(self, X):
        """
        Make predictions with the trained model.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
        
        Returns:
        --------
        numpy.ndarray
            Predictions
        """
        return self.forward(X)
    
    def save(self, filename):
        """
        Save the model to a file.
        
        Parameters:
        -----------
        filename : str
            Filename to save the model to
        """
        model_data = {
            'layer_sizes': self.layer_sizes,
            'activation_functions': self.activation_functions,
            'loss_function': self.loss_function,
            'weights': self.weights,
            'biases': self.biases,
            'weight_init_method': self.weight_init_method,
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound,
            'seed': self.seed
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filename):
        """
        Load model from a file.
        
        Parameters:
        -----------
        filename : str
            Filename to load the model from
        
        Returns:
        --------
        FFNN
            Loaded model
        """
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(
            model_data['layer_sizes'],
            model_data['activation_functions'],
            model_data['loss_function'],
            model_data['weight_init_method'],
            model_data['lower_bound'],
            model_data['upper_bound'],
            model_data['seed']
        )
        
        model.weights = model_data['weights']
        model.biases = model_data['biases']
        
        # Initialize gradients
        model.weight_gradients = [np.zeros_like(w) for w in model.weights]
        model.bias_gradients = [np.zeros_like(b) for b in model.biases]
        
        return model
    
    def visualize_model(self, figsize=(12, 8)):
        """
        Visualize the model as a graph with weights and gradients.
        """
        plt.figure(figsize=figsize)
        G = nx.DiGraph()
        
        # Add nodes for each layer
        node_positions = {}
        max_layer_size = max(self.layer_sizes)
        
        # Create nodes for each neuron
        for layer_idx, layer_size in enumerate(self.layer_sizes):
            layer_name = "Input" if layer_idx == 0 else "Output" if layer_idx == len(self.layer_sizes) - 1 else f"Hidden {layer_idx}"
            
            # Calculate vertical positions for neurons in this layer
            vertical_spacing = max_layer_size * 0.8 / max(layer_size, 1)
            vertical_start = -(layer_size - 1) * vertical_spacing / 2
            
            for neuron_idx in range(layer_size):
                node_id = f"L{layer_idx}N{neuron_idx}"
                G.add_node(node_id, layer=layer_idx, neuron=neuron_idx)
                node_positions[node_id] = (layer_idx * 2, vertical_start + neuron_idx * vertical_spacing)
        
        # Add edges with weight and gradient information
        for layer_idx in range(len(self.layer_sizes) - 1):
            # Get activation function for this layer
            activation = self.activation_functions[layer_idx + 1]
            
            for i in range(self.layer_sizes[layer_idx]):
                for j in range(self.layer_sizes[layer_idx + 1]):
                    source_id = f"L{layer_idx}N{i}"
                    target_id = f"L{layer_idx + 1}N{j}"
                    weight = self.weights[layer_idx][i, j]
                    weight_gradient = self.weight_gradients[layer_idx][i, j]
                    
                    G.add_edge(
                        source_id, 
                        target_id, 
                        weight=weight,
                        gradient=weight_gradient,
                        activation=activation
                    )
        
        # Draw the network
        pos = node_positions
        
        # Draw nodes
        for layer_idx in range(len(self.layer_sizes)):
            layer_nodes = [n for n, attr in G.nodes(data=True) if attr['layer'] == layer_idx]
            layer_name = "Input" if layer_idx == 0 else "Output" if layer_idx == len(self.layer_sizes) - 1 else f"Hidden {layer_idx}"
            
            nx.draw_networkx_nodes(
                G, pos, 
                nodelist=layer_nodes,
                node_color=f"C{layer_idx}", 
                node_size=500, 
                alpha=0.8,
                label=f"{layer_name} Layer"
            )
        
        # Draw edges with different colors based on weight values
        edges = G.edges(data=True)
        weights = [abs(d['weight']) for _, _, d in edges]
        max_weight = max(weights) if weights else 1
        
        # Normalize weights for visual representation
        normalized_weights = [2 + 3 * (w / max_weight) for w in weights]
        
        # Get edge colors based on sign of weight
        edge_colors = ['red' if d['weight'] < 0 else 'green' for _, _, d in edges]
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos, 
            width=normalized_weights,
            edge_color=edge_colors,
            alpha=0.6,
            arrows=True,
            arrowsize=15
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            G, pos, 
            labels={node: f"{node.split('N')[1]}" for node in G.nodes()},
            font_size=10,
            font_color='white'
        )
        
        # Add activation functions to the plot
        for layer_idx in range(len(self.activation_functions)):
            plt.text(
                layer_idx * 2 + 0.5, 
                max(node_positions.values(), key=lambda x: x[1])[1] + 1,
                f"Activation: {self.activation_functions[layer_idx]}",
                horizontalalignment='center',
                fontsize=10
            )
        
        plt.title("Neural Network Structure", fontsize=15)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(self.layer_sizes))
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def plot_weight_distribution(self, layers=None, figsize=(12, 5)):
        """
        Plot the distribution of weights for specified layers.
        
        Parameters:
        -----------
        layers : list or None
            List of layer indices to plot. If None, plot all layers.
        figsize : tuple
            Figure size
        """
        if layers is None:
            layers = list(range(len(self.weights)))
        
        plt.figure(figsize=figsize)
        
        for i, layer_idx in enumerate(layers):
            if layer_idx < 0 or layer_idx >= len(self.weights):
                print(f"Warning: Layer {layer_idx} does not exist. Skipping.")
                continue
            
            weights = self.weights[layer_idx].flatten()
            plt.subplot(1, len(layers), i + 1)
            plt.hist(weights, bins=30, alpha=0.7, label=f'Layer {layer_idx}')
            plt.title(f"Layer {layer_idx}-{layer_idx+1} Weights")
            plt.xlabel("Weight Value")
            plt.ylabel("Frequency")
            plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_gradient_distribution(self, layers=None, figsize=(12, 5)):
        """
        Plot the distribution of weight gradients for specified layers.
        
        Parameters:
        -----------
        layers : list or None
            List of layer indices to plot. If None, plot all layers.
        figsize : tuple
            Figure size
        """
        if layers is None:
            layers = list(range(len(self.weight_gradients)))
        
        plt.figure(figsize=figsize)
        
        for i, layer_idx in enumerate(layers):
            if layer_idx < 0 or layer_idx >= len(self.weight_gradients):
                print(f"Warning: Layer {layer_idx} does not exist. Skipping.")
                continue
            
            gradients = self.weight_gradients[layer_idx].flatten()
            plt.subplot(1, len(layers), i + 1)
            plt.hist(gradients, bins=30, alpha=0.7, label=f'Layer {layer_idx}')
            plt.title(f"Layer {layer_idx}-{layer_idx+1} Gradients")
            plt.xlabel("Gradient Value")
            plt.ylabel("Frequency")
            plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()