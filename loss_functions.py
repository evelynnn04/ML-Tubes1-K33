import numpy as np

def binary_cross_entropy(y_true, y_pred):
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

def binary_cross_entropy_derivative(y_true, y_pred):
    """Derivative of binary cross-entropy loss with respect to y_pred"""
    # Assuming sigmoid activation is used for output layer
    batch_size = y_true.shape[0]
    return (y_pred - y_true) / batch_size

# Dictionary to map loss function names to functions
LOSS_FUNCTIONS = {
    'binary_cross_entropy': binary_cross_entropy
}

# Dictionary to map loss function names to their derivatives
LOSS_DERIVATIVES = {
    'binary_cross_entropy': binary_cross_entropy_derivative
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