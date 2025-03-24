import numpy as np

def linear(x):
    """Linear activation function (identity)"""
    return x

def linear_derivative(x):
    """Derivative of linear function"""
    return np.ones_like(x)

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU function"""
    return np.where(x > 0, 1, 0)

def tanh(x):
    """Hyperbolic tangent activation function"""
    return np.tanh(x)

def tanh_derivative(x):
    """Derivative of hyperbolic tangent function"""
    return 1 - np.tanh(x) ** 2

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow

def sigmoid_derivative(x):
    """Derivative of sigmoid function"""
    s = sigmoid(x)
    return s * (1 - s)

def apply_activation(x, activation_name):
    """Apply activation function to input"""
    if activation_name == 'linear':
        return linear(x)
    elif activation_name == 'relu':
        return relu(x)
    elif activation_name == 'tanh':
        return tanh(x)
    elif activation_name == 'sigmoid':
        return sigmoid(x)
    else:
        raise ValueError(f"Activation function {activation_name} not supported.")

def apply_activation_derivative(x, activation_name):
    """Apply derivative of activation function to input"""
    if activation_name == 'linear':
        return linear_derivative(x)
    elif activation_name == 'relu':
        return relu_derivative(x)
    elif activation_name == 'tanh':
        return tanh_derivative(x)
    elif activation_name == 'sigmoid':
        return sigmoid_derivative(x)
    else:
        raise ValueError(f"Activation function {activation_name} not supported.")