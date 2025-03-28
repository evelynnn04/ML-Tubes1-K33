import numpy as np

def mean_squared_error(y_true, y_pred):
    """Mean Squared Error (MSE) loss function"""
    return np.mean(np.square(y_true - y_pred))

def mean_squared_error_derivative(y_true, y_pred):
    """Derivative of Mean Squared Error loss"""
    return 2 * (y_pred - y_true) / y_true.shape[0]

def binary_cross_entropy(y_true, y_pred):
    """Binary Cross-Entropy loss function"""
    # Clip predictions to avoid log(0)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # Handle binary classification
    if y_true.shape[1] == 1:
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        # Handle multi-class case with binary cross-entropy
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    return loss

def binary_cross_entropy_derivative(y_true, y_pred):
    """Derivative of Binary Cross-Entropy loss"""
    # Assuming sigmoid activation is used for output layer
    batch_size = y_true.shape[0]
    return (y_pred - y_true) / batch_size

def categorical_cross_entropy(y_true, y_pred):
    """Categorical Cross-Entropy loss function"""
    # Clip predictions to avoid log(0)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # Compute categorical cross-entropy
    loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    return loss

def categorical_cross_entropy_derivative(y_true, y_pred):
    """Derivative of Categorical Cross-Entropy loss"""
    batch_size = y_true.shape[0]
    return (y_pred - y_true) / batch_size

# Dictionary to map loss function names to functions
LOSS_FUNCTIONS = {
    'mse': mean_squared_error,
    'binary_cross_entropy': binary_cross_entropy,
    'categorical_cross_entropy': categorical_cross_entropy
}

# Dictionary to map loss function names to their derivatives
LOSS_DERIVATIVES = {
    'mse': mean_squared_error_derivative,
    'binary_cross_entropy': binary_cross_entropy_derivative,
    'categorical_cross_entropy': categorical_cross_entropy_derivative
}

def get_loss_function(loss_name):
    """Get loss function by name"""
    if loss_name in LOSS_FUNCTIONS:
        return LOSS_FUNCTIONS[loss_name]
    else:
        raise ValueError(f"Loss function {loss_name} not supported.")

def get_loss_derivative(loss_name):
    """Get loss derivative function by name"""
    if loss_name in LOSS_DERIVATIVES:
        return LOSS_DERIVATIVES[loss_name]
    else:
        raise ValueError(f"Loss derivative for {loss_name} not supported.")