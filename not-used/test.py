import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import time

from ffnn import FFNN

def test_binary_classification():
    """Test FFNN on a binary classification task using MNIST (digit 0 vs. not 0)"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    print("Dataset loaded!")
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Convert labels to binary (for digit classification)
    # We'll classify digits as either 0 or not 0
    y_binary = np.array([1.0 if label == '0' else 0.0 for label in y]).reshape(-1, 1)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)
    
    # Create a smaller subset for faster training during testing
    X_train_small = X_train[:5000]
    y_train_small = y_train[:5000]
    X_test_small = X_test[:1000]
    y_test_small = y_test[:1000]
    
    # Define neural network architecture
    input_size = X.shape[1]  # 784 features (28x28 pixels)
    hidden_size = 128  # Size of hidden layer
    output_size = 1  # Binary classification: 0 or not 0
    
    # Create model
    model = FFNN(
        layer_sizes=[input_size, hidden_size, output_size],
        activation_functions=['linear', 'relu', 'sigmoid'],  # Note: input layer uses linear activation
        loss_function='binary_cross_entropy',
        weight_init_method='uniform',
        lower_bound=-0.1,
        upper_bound=0.1,
        seed=42
    )
    
    # Display model structure
    # print("Model structure before training:")
    # model.visualize_model()
    
    # # Display initial weight distribution
    # print("Initial weight distribution:")
    # model.plot_weight_distribution()
    
    # Train the model
    print("Training the model...")
    start_time = time.time()
    history = model.fit(
        X_train_small, 
        y_train_small, 
        X_val=X_test_small, 
        y_val=y_test_small,
        batch_size=32,
        learning_rate=0.01,
        epochs=10,
        verbose=1
    )
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")
    
    # # Display weight distribution after training
    # print("Weight distribution after training:")
    # model.plot_weight_distribution()
    
    # # Display gradient distribution after training
    # print("Gradient distribution after training:")
    # model.plot_gradient_distribution()
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Evaluate the model
    y_pred = model.predict(X_test_small)
    y_pred_binary = (y_pred > 0.5).astype(float)
    accuracy = np.mean(y_pred_binary == y_test_small)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Save the model
    model.save('mnist_binary_model.pkl')
    print("Model saved as 'mnist_binary_model.pkl'")
    
    # Load the model and verify
    loaded_model = FFNN.load('mnist_binary_model.pkl')
    y_pred_loaded = loaded_model.predict(X_test_small)
    y_pred_loaded_binary = (y_pred_loaded > 0.5).astype(float)
    accuracy_loaded = np.mean(y_pred_loaded_binary == y_test_small)
    print(f"Test accuracy (loaded model): {accuracy_loaded:.4f}")
    
    # Visualize some predictions
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(10):
        # Get a random sample
        idx = np.random.randint(0, len(X_test_small))
        img = X_test_small[idx].reshape(28, 28)
        true_label = "0" if y_test_small[idx, 0] == 1 else "Not 0"
        pred_label = "0" if y_pred_binary[idx, 0] == 1 else "Not 0"
        
        # Display the image
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def test_multiclass_classification():
    """Test FFNN on a multiclass classification task using MNIST (all 10 digits)"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    print("Dataset loaded!")
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # One-hot encode labels
    lb = LabelBinarizer()
    y_one_hot = lb.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
    
    # Create a smaller subset for faster training
    X_train_small = X_train[:5000]
    y_train_small = y_train[:5000]
    X_test_small = X_test[:1000]
    y_test_small = y_test[:1000]
    
    # Define model architecture
    input_size = X.shape[1]  # 784 features
    hidden_size1 = 128
    hidden_size2 = 64
    output_size = 10  # 10 classes (digits 0-9)
    
    # Create model with 2 hidden layers
    model = FFNN(
        layer_sizes=[input_size, hidden_size1, hidden_size2, output_size],
        activation_functions=['linear', 'relu', 'relu', 'sigmoid'],
        loss_function='binary_cross_entropy',
        weight_init_method='uniform',
        lower_bound=-0.05,
        upper_bound=0.05,
        seed=42
    )
    
    # Visualize initial model
    print("Model structure before training:")
    model.visualize_model()
    
    # Train model
    print("Training the model...")
    history = model.fit(
        X_train_small,
        y_train_small,
        X_val=X_test_small,
        y_val=y_test_small,
        batch_size=32,
        learning_rate=0.01,
        epochs=20,
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Evaluate model
    y_pred = model.predict(X_test_small)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test_small, axis=1)
    accuracy = np.mean(y_pred_classes == y_true_classes)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Save model
    model.save('mnist_multiclass_model.pkl')
    print("Model saved as 'mnist_multiclass_model.pkl'")

if __name__ == "__main__": test_binary_classification()