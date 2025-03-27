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

def softmax(x):
    """Softmax activation function"""
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def softmax_derivative(x):
    """Derivative of softmax function"""
    # Compute softmax values
    s = softmax(x)
    
    # Create diagonal matrices and subtract
    jacobian_matrix = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        # Create diagonal of softmax output
        diag = np.diag(s[i])
        # Subtract outer product
        jacobian_matrix[i] = diag - np.outer(s[i], s[i])
    
    return jacobian_matrix

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1, alpha * np.exp(x))

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
    elif activation_name == 'softmax':
        return softmax(x)
    elif activation_name == 'leaky_relu':
        return leaky_relu(x)
    elif activation_name == 'elu':
        return elu(x)
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
    elif activation_name == 'softmax':
        return softmax_derivative(x)
    elif activation_name == 'leaky_relu':
        return leaky_relu_derivative(x)
    elif activation_name == 'elu':
        return elu_derivative(x)
    else:
        raise ValueError(f"Activation function {activation_name} not supported.")
    

'''
References:
https://medium.com/@juanc.olamendy/understanding-relu-leakyrelu-and-prelu-a-comprehensive-guide-20f2775d3d64
'''