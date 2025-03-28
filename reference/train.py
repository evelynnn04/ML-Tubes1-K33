import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import time

from ffnn import FFNN
from visualization import visualize_predictions, plot_training_history

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

    # model_zero = FFNN(
    #     layer_sizes=[784, 128, 10],
    #     activation_functions=['relu', 'sigmoid'],
    #     loss_function='binary_cross_entropy',
    #     weight_init_method='zero'
    # )

    # model_normal = FFNN(
    #     layer_sizes=[784, 128, 10],
    #     activation_functions=['relu', 'sigmoid'],
    #     loss_function='binary_cross_entropy',
    #     weight_init_method='normal',
    #     mean=0.0,  # default mean
    #     variance=0.1  # default variance
    # )
    
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
    
    # Plot training history
    plot_training_history(history)
    
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
    visualize_predictions(X_test_small, y_test_small, y_pred_binary, n_samples=10)

def test_multiclass_classification():
    """Test FFNN on a multi-class classification task using MNIST (digits 0-9)"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    print("Dataset loaded!")
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Convert labels to one-hot encoded format
    lb = LabelBinarizer()
    y_onehot = lb.fit_transform(y)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
    
    # Create a smaller subset for faster training during testing
    X_train_small = X_train[:5000]
    y_train_small = y_train[:5000]
    X_test_small = X_test[:1000]
    y_test_small = y_test[:1000]
    
    # Define neural network architecture
    input_size = X.shape[1]  # 784 features (28x28 pixels)
    hidden_size = 128  # Size of hidden layer
    output_size = 10  # Multi-class classification: 10 classes (digits 0-9)
    
    # Create model
    # model = FFNN(
    #     layer_sizes=[input_size, hidden_size, output_size],
    #     activation_functions=['linear', 'relu', 'sigmoid'],  # Note: input layer uses linear activation
    #     loss_function='binary_cross_entropy',
    #     weight_init_method='uniform',
    #     lower_bound=-0.1,
    #     upper_bound=0.1,
    #     seed=42
    # )
    model = FFNN(
        layer_sizes=[784, 128, 10],
        activation_functions=['relu', 'sigmoid'],
        loss_function='binary_cross_entropy',
        weight_init_method='zero'
    )
    
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
    
    # Plot
    plot_training_history(history)

    # Evaluate the model
    y_pred = model.predict(X_test_small)
    y_pred_labels = lb.inverse_transform(y_pred)
    y_true_labels = lb.inverse_transform(y_test_small)
    accuracy = np.mean(y_pred_labels == y_true_labels)
    print(f"Test accuracy: {accuracy:.4f}")

    # Save the model
    model.save('mnist_multiclass_model.pkl')
    print("Model saved as 'mnist_multiclass_model.pkl'")

    # Load the model and verify
    loaded_model = FFNN.load('mnist_multiclass_model.pkl')
    y_pred_loaded = loaded_model.predict(X_test_small)
    y_pred_loaded_labels = lb.inverse_transform(y_pred_loaded)
    accuracy_loaded = np.mean(y_pred_loaded_labels == y_true_labels)
    print(f"Test accuracy (loaded model): {accuracy_loaded:.4f}")

    # Visualize some predictions
    visualize_predictions(X_test_small, y_test_small, y_pred, n_samples=10, is_binary=False)

if __name__ == "__main__":
    # test_binary_classification()
    test_multiclass_classification()