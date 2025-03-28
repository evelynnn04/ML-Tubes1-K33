import numpy as np
import pickle
from tqdm import tqdm

# Import custom modules
from activation_functions import apply_activation, apply_activation_derivative, sigmoid
from loss_functions import get_loss_function, get_loss_derivative
import visualization as vis

class FFNN:
    def __init__(self, layer_sizes, activation_functions, loss_function, 
                 weight_init_method='uniform', 
                 lower_bound=-0.1, upper_bound=0.1,  # For uniform
                 mean=0.0, variance=0.1,  # For normal
                 seed=42):
        """
        Initialize a Feedforward Neural Network with enhanced weight initialization.
        
        Parameters:
        -----------
        layer_sizes : list
            Number of neurons in each layer (including input and output layers)
        activation_functions : list
            Activation function for each layer (including input layer)
        loss_function : str
            Loss function to use
        weight_init_method : str
            Method to initialize weights 
            Options: 'uniform', 'normal', 'zero'
        lower_bound : float
            Lower bound for uniform weight initialization
        upper_bound : float
            Upper bound for uniform weight initialization
        mean : float
            Mean for normal distribution weight initialization
        variance : float
            Variance for normal distribution weight initialization
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
        self.mean = mean
        self.variance = variance
        self.seed = seed
        
        # Initialize weights and biases
        np.random.seed(seed)
        self.weights = []
        self.biases = []
        self.weight_gradients = []
        self.bias_gradients = []
        
        # Initialize weights between layers
        for i in range(len(layer_sizes) - 1):
            # Weight initialization based on method
            if weight_init_method == 'uniform':
                w = np.random.uniform(lower_bound, upper_bound, (layer_sizes[i], layer_sizes[i+1]))
                b = np.random.uniform(lower_bound, upper_bound, (1, layer_sizes[i+1]))
            elif weight_init_method == 'normal':
                # Use normal distribution with specified mean and variance
                w = np.random.normal(loc=mean, scale=np.sqrt(variance), 
                                     size=(layer_sizes[i], layer_sizes[i+1]))
                b = np.random.normal(loc=mean, scale=np.sqrt(variance), 
                                     size=(1, layer_sizes[i+1]))
            elif weight_init_method == 'zero':
                # Initialize weights and biases to zero
                w = np.zeros((layer_sizes[i], layer_sizes[i+1]))
                b = np.zeros((1, layer_sizes[i+1]))
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
                a = sigmoid(z)
            else:
                # For other layers, use the specified activation function
                a = apply_activation(z, self.activation_functions[i+1])
            
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
        
        # Get loss derivative
        loss_derivative = get_loss_derivative(self.loss_function)
        
        # Initialize delta for output layer
        delta = loss_derivative(y_true, self.activations[-1])
        
        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Calculate gradients for current layer
            self.weight_gradients[i] = np.dot(self.activations[i].T, delta)
            self.bias_gradients[i] = np.sum(delta, axis=0, keepdims=True)
            
            # Calculate delta for previous layer (if not at input layer)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * apply_activation_derivative(
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
        
        # Get loss function
        loss_func = get_loss_function(self.loss_function)
        
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
                batch_loss = loss_func(y_batch, y_pred)
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
                val_loss = loss_func(y_val, val_pred)
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
    
    # Visualization methods that delegate to the visualization module
    def visualize_model(self, figsize=(12, 8)):
        """Visualize the model structure"""
        vis.visualize_model(self, figsize)
    
    def plot_weight_distribution(self, layers=None, figsize=(12, 5)):
        """Plot weight distribution"""
        vis.plot_weight_distribution(self, layers, figsize)
    
    def plot_gradient_distribution(self, layers=None, figsize=(12, 5)):
        """Plot gradient distribution"""
        vis.plot_gradient_distribution(self, layers, figsize)
    
    def plot_training_history(self, figsize=(10, 6)):
        """Plot training history"""
        vis.plot_training_history(self.history, figsize)