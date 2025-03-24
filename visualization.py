import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def visualize_model(model, figsize=(12, 8)):
    """
    Visualize the model as a graph with weights and gradients.
    
    Parameters:
    -----------
    model : FFNN
        Neural network model to visualize
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)
    G = nx.DiGraph()
    
    # Add nodes for each layer
    node_positions = {}
    max_layer_size = max(model.layer_sizes)
    
    # Create nodes for each neuron
    for layer_idx, layer_size in enumerate(model.layer_sizes):
        layer_name = "Input" if layer_idx == 0 else "Output" if layer_idx == len(model.layer_sizes) - 1 else f"Hidden {layer_idx}"
        
        # Calculate vertical positions for neurons in this layer
        vertical_spacing = max_layer_size * 0.8 / max(layer_size, 1)
        vertical_start = -(layer_size - 1) * vertical_spacing / 2
        
        for neuron_idx in range(layer_size):
            node_id = f"L{layer_idx}N{neuron_idx}"
            G.add_node(node_id, layer=layer_idx, neuron=neuron_idx)
            node_positions[node_id] = (layer_idx * 2, vertical_start + neuron_idx * vertical_spacing)
    
    # Add edges with weight and gradient information
    for layer_idx in range(len(model.layer_sizes) - 1):
        # Get activation function for this layer
        activation = model.activation_functions[layer_idx + 1]
        
        for i in range(model.layer_sizes[layer_idx]):
            for j in range(model.layer_sizes[layer_idx + 1]):
                source_id = f"L{layer_idx}N{i}"
                target_id = f"L{layer_idx + 1}N{j}"
                weight = model.weights[layer_idx][i, j]
                weight_gradient = model.weight_gradients[layer_idx][i, j]
                
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
    for layer_idx in range(len(model.layer_sizes)):
        layer_nodes = [n for n, attr in G.nodes(data=True) if attr['layer'] == layer_idx]
        layer_name = "Input" if layer_idx == 0 else "Output" if layer_idx == len(model.layer_sizes) - 1 else f"Hidden {layer_idx}"
        
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
    for layer_idx in range(len(model.activation_functions)):
        plt.text(
            layer_idx * 2 + 0.5, 
            max(node_positions.values(), key=lambda x: x[1])[1] + 1,
            f"Activation: {model.activation_functions[layer_idx]}",
            horizontalalignment='center',
            fontsize=10
        )
    
    plt.title("Neural Network Structure", fontsize=15)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(model.layer_sizes))
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_weight_distribution(model, layers=None, figsize=(12, 5)):
    """
    Plot the distribution of weights for specified layers.
    
    Parameters:
    -----------
    model : FFNN
        Neural network model
    layers : list or None
        List of layer indices to plot. If None, plot all layers.
    figsize : tuple
        Figure size
    """
    if layers is None:
        layers = list(range(len(model.weights)))
    
    plt.figure(figsize=figsize)
    
    for i, layer_idx in enumerate(layers):
        if layer_idx < 0 or layer_idx >= len(model.weights):
            print(f"Warning: Layer {layer_idx} does not exist. Skipping.")
            continue
        
        weights = model.weights[layer_idx].flatten()
        plt.subplot(1, len(layers), i + 1)
        plt.hist(weights, bins=30, alpha=0.7, label=f'Layer {layer_idx}')
        plt.title(f"Layer {layer_idx}-{layer_idx+1} Weights")
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_gradient_distribution(model, layers=None, figsize=(12, 5)):
    """
    Plot the distribution of weight gradients for specified layers.
    
    Parameters:
    -----------
    model : FFNN
        Neural network model
    layers : list or None
        List of layer indices to plot. If None, plot all layers.
    figsize : tuple
        Figure size
    """
    if layers is None:
        layers = list(range(len(model.weight_gradients)))
    
    plt.figure(figsize=figsize)
    
    for i, layer_idx in enumerate(layers):
        if layer_idx < 0 or layer_idx >= len(model.weight_gradients):
            print(f"Warning: Layer {layer_idx} does not exist. Skipping.")
            continue
        
        gradients = model.weight_gradients[layer_idx].flatten()
        plt.subplot(1, len(layers), i + 1)
        plt.hist(gradients, bins=30, alpha=0.7, label=f'Layer {layer_idx}')
        plt.title(f"Layer {layer_idx}-{layer_idx+1} Gradients")
        plt.xlabel("Gradient Value")
        plt.ylabel("Frequency")
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_training_history(history, figsize=(10, 6)):
    """
    Plot the training history.
    
    Parameters:
    -----------
    history : dict
        Training history containing train_loss and val_loss
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)
    plt.plot(history['train_loss'], label='Training Loss')
    if 'val_loss' in history and history['val_loss']:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def visualize_predictions(X_test, y_test, y_pred, n_samples=10, figsize=(15, 6), is_binary=True):
    """
    Visualize predictions for image data (e.g., MNIST).
    
    Parameters:
    -----------
    X_test : numpy.ndarray
        Test data
    y_test : numpy.ndarray
        True labels
    y_pred : numpy.ndarray
        Predicted labels
    n_samples : int
        Number of samples to visualize
    figsize : tuple
        Figure size
    is_binary : bool
        Whether the classification is binary or multiclass
    """
    fig, axes = plt.subplots(2, n_samples//2, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(n_samples):
        # Get a random sample
        idx = np.random.randint(0, len(X_test))
        img = X_test[idx].reshape(28, 28)  # Assuming 28x28 images (MNIST)
        
        if is_binary:
            true_label = "0" if y_test[idx, 0] == 1 else "Not 0"
            pred_label = "0" if y_pred[idx, 0] > 0.5 else "Not 0"
        else:
            true_label = str(np.argmax(y_test[idx]))
            pred_label = str(np.argmax(y_pred[idx]))
        
        # Display the image
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()