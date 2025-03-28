import sys  
import time  
import math  
import uuid  
import numpy as np  
import matplotlib.pyplot as plt
import pickle
import plotly.graph_objs as go
from numpy import errstate  
from sklearn.preprocessing import OneHotEncoder  
from tqdm import tqdm

sys.setrecursionlimit(10000)

from varValue import VarValue
from layer import Layer

class FFNN:
    def __init__(self, loss='mse', batch_size=1, learning_rate=0.01, epochs=20, verbose=0):
        self.loss = loss    # mse/bce/cce
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.verbose=verbose
        self.layers = None
        self.weights = []
        self.bias = []
        self.x = None
        self.y = None
        self.onehot_encoder = OneHotEncoder(categories='auto')

    def __loss(self, out, target):
        if out.shape != target.shape:
            print("Output shape: ", out.shape)
            print("Target shape: ", target.shape)
            raise ValueError("Shape not match")

        if self.loss == 'mse':
            mse = (1/target.shape[1]) * np.square(target - out)
            return np.sum(mse)

        elif self.loss == 'bce':
            epsilon = 1e-15
            # loss = VarValue(0.0, varname='loss_bce')
            loss = 0

            for i in range(target.shape[0]):            # loop untuk batch
                for j in range(target.shape[1]):        # loop untuk neuron output
                    # target_val = VarValue(target[i, j], varname=f'target_{i}_{j}')

                    # Clipping untuk mencegah log(0)
                    out[i, j].value = np.clip(out[i, j].value, epsilon, 1 - epsilon)

                    # BCE: -(y*log(p) + (1-y)*log(1-p))
                    term1 = target * out[i, j].log()
                    term2 = (1 - target) * (1 - out[i, j]).log()

                    loss = loss + (term1 + term2)

            loss = loss * (-1 / self.batch_size)
            return loss

        elif self.loss == 'cce':
            epsilon = 1e-15
            loss = VarValue(0.0, varname='loss_const')

            for i in range(target.shape[0]):            # loop untuk batch
                for j in range(target.shape[1]):        # loop untuk neuron output
                    target_val = VarValue(target[i, j], varname=f'target_{i}_{j}')
                    out[i, j].value = np.clip(out[i, j].value, epsilon, 1 - epsilon)
                    loss = loss + (target_val * out[i, j].log())

            loss = loss * (-1 / self.batch_size)
            return loss

    def build_layers(self, *layers: Layer):
        self.layers = layers
        for layer in self.layers:
            layer.learning_rate = self.learning_rate

    def fit(self, x, y, validation_data=None):
        self.x = x
        self.y = self.onehot_encoder.fit_transform(y.reshape(-1, 1)).toarray()
        
        history = {'train_loss': [], 'val_loss': []}
        
        total_batch = (len(x) + self.batch_size - 1) // self.batch_size
        start_global = time.time()

        epoch_iterator = tqdm(range(self.epochs), desc="Training Progress", unit="epoch") if self.verbose else range(self.epochs)

        for epoch in epoch_iterator:
            train_loss_epoch = 0.0

            batch_iterator = tqdm(range(total_batch), desc=f"Epoch {epoch + 1}", unit="batch", leave=False) if self.verbose else range(total_batch)

            for i in batch_iterator:
                x_batch = self.x[i * self.batch_size:(i + 1) * self.batch_size] if (i + 1) < total_batch else self.x[i * self.batch_size:]
                y_batch = self.y[i * self.batch_size:(i + 1) * self.batch_size] if (i + 1) < total_batch else self.y[i * self.batch_size:]

                # Forward pass
                batch_input = x_batch
                for layer in self.layers:
                    layer.forward(batch_input)
                    batch_input = layer.out
                out = batch_input

                # Hitung loss
                error = self.__loss(out, y_batch)
                train_loss_epoch += error.value if isinstance(error, VarValue) else error

                # Backward pass
                for layer in reversed(self.layers):
                    layer.backward(err=error)

                for layer in self.layers:
                    layer.clean_derivative()

            # Rata-rata training loss per epoch
            avg_train_loss = train_loss_epoch / total_batch
            history['train_loss'].append(avg_train_loss)

            # Jika ada data validasi, hitung validation loss
            if validation_data:
                X_val, y_val = validation_data
                y_val_enc = self.onehot_encoder.transform(y_val.reshape(-1, 1)).toarray()
                val_pred = self.predict(X_val)
                val_loss = self.__loss(val_pred, y_val_enc)
                avg_val_loss = val_loss.value if isinstance(val_loss, VarValue) else val_loss
                history['val_loss'].append(avg_val_loss)

            if self.verbose:
                print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}", end="")
                if validation_data:
                    print(f", Validation Loss: {avg_val_loss:.4f}")
                else:
                    print()

        end_global = time.time()
        print("Total Duration:", end_global - start_global)

        return history

    def predict(self, x_predict):
        batch_input = x_predict
        for layer in self.layers:
            layer.forward(batch_input)
            batch_input = layer.out
        out = batch_input
        return out
        
    def visualize(self):
        num_layers = len(self.layers)

        # Warna untuk setiap layer
        layer_colors = {
            0: 'yellow',    # Input layer
            -1: 'salmon',   # Output layer
            'hidden': 'lightblue'  # Hidden layers
        }

        nodes_x, nodes_y, node_colors, node_texts = [], [], [], []

        # Membuat koordinat dan informasi node
        for layer_idx, layer in enumerate(self.layers):
            if num_layers == 1:
                color = layer_colors[-1]
            elif layer_idx == 0:
                color = layer_colors[0]
            elif layer_idx == num_layers - 1:
                color = layer_colors[-1]
            else:
                color = layer_colors['hidden']

            n_neurons = layer.n_neurons
            y_positions = np.linspace(0, 1, n_neurons)
            x_pos = layer_idx / (num_layers - 1) if num_layers > 1 else 0.5

            for neuron_idx, y_pos in enumerate(y_positions):
                nodes_x.append(x_pos)
                nodes_y.append(y_pos)
                node_colors.append(color)

                node_info = f"Layer {layer_idx}, Neuron {neuron_idx}<br>Activation: {layer.activation}"

                # Menampilkan gradient dari bias jika tersedia
                if layer.grad_biases is not None:
                    node_info += f"<br>Bias Gradient: {layer.grad_biases[neuron_idx]:.4f}"
                else:
                    node_info += "<br>Bias Gradient: N/A"

                node_texts.append(node_info)

        # Membuat node scatter plot
        node_trace = go.Scatter(
            x=nodes_x, y=nodes_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=False,
                color=node_colors,
                size=15,
                line_width=2
            ),
            text=node_texts
        )

        # Membuat edge scatter plot terpisah untuk tiap koneksi (agar hoverable)
        edge_traces = []

        for layer_idx, layer in enumerate(self.layers[:-1]):
            next_layer = self.layers[layer_idx + 1]
            x_pos = layer_idx / (num_layers - 1)
            next_x = (layer_idx + 1) / (num_layers - 1)

            y_positions = np.linspace(0, 1, layer.n_neurons)
            next_y_positions = np.linspace(0, 1, next_layer.n_neurons)

            for curr_neuron_idx, curr_y in enumerate(y_positions):
                for next_neuron_idx, next_y in enumerate(next_y_positions):

                    # Informasi bobot dan gradient bobot
                    weight_text = "Weight: Not initialized"
                    gradient_text = "Gradient: N/A"

                    if layer.weights is not None:
                        try:
                            weight = next_layer.weights[curr_neuron_idx][next_neuron_idx].value
                            weight_text = f"Weight: {weight:.4f}"
                        except:
                            weight_text = "Weight: Unavailable"

                    if layer.grad_weights is not None:
                        try:
                            gradient = next_layer.grad_weights[curr_neuron_idx][next_neuron_idx]
                            gradient_text = f"Gradient: {gradient:.4f}"
                        except:
                            gradient_text = "Gradient: Unavailable"

                    edge_trace = go.Scatter(
                        x=[x_pos, next_x],
                        y=[curr_y, next_y],
                        mode='lines+markers',
                        line=dict(width=0.5, color='#888'),
                        hoverinfo='text',
                        text=f"{weight_text}<br>{gradient_text}",
                        marker=dict(size=10, opacity=0) # marker transparan tapi hoverable
                    )

                    edge_traces.append(edge_trace)

        # Buat figure dengan nodes dan edges
        fig = go.Figure(data=edge_traces + [node_trace],
                        layout=go.Layout(
                            title='Neural Network Architecture',
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=0,l=0,r=0,t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            plot_bgcolor='rgba(255,255,255,255)'
                        ))
        fig.show()


    
    def save(self, filename):
        model_data = {
            'loss': self.loss,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'layers': [
                {'n_neurons': layer.n_neurons, 'activation': layer.activation, 'weights': layer.weights.tolist(), 'biases': layer.biases.tolist()}
                for layer in self.layers
            ]
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filename}")

    def load(self, filename):
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        self.loss = model_data['loss']
        self.batch_size = model_data['batch_size']
        self.learning_rate = model_data['learning_rate']
        self.epochs = model_data['epochs']
        self.layers = [
            Layer(n_neurons=layer['n_neurons'], activation=layer['activation'], weights=np.array(layer['weights']), biases=np.array(layer['biases']))
            for layer in model_data['layers']
        ]
        print(f"Model loaded from {filename}")

    def plot_weights_distribution(self, layers_to_plot):
        num_layers = len(layers_to_plot)
        plt.figure(figsize=(6*num_layers, 4))

        for idx, layer_idx in enumerate(layers_to_plot):
            layer = self.layers[layer_idx]
            weights_values = [w.value for neuron_weights in layer.weights for w in neuron_weights]

            plt.subplot(1, num_layers, idx + 1)
            plt.boxplot(weights_values, vert=True, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', color='black'),
                        medianprops=dict(color='black'))
            plt.title(f'Distribusi Bobot Layer {layer_idx+1}')
            plt.ylabel('Nilai Bobot')
            plt.xticks([])

        plt.tight_layout()
        plt.show()

    def plot_gradients_distribution(self, layers_to_plot):
        num_layers = len(layers_to_plot)
        plt.figure(figsize=(6*num_layers, 4))

        for idx, layer_idx in enumerate(layers_to_plot):
            layer = self.layers[layer_idx]
            gradients_values = [gw for grad_weights in layer.grad_weights for gw in grad_weights]

            plt.subplot(1, num_layers, idx + 1)
            plt.boxplot(gradients_values, vert=True, patch_artist=True,
                        boxprops=dict(facecolor='lightpink', color='black'),
                        medianprops=dict(color='black'))
            plt.title(f'Distribusi Gradien Layer {layer_idx+1}')
            plt.ylabel('Nilai Gradien')
            plt.xticks([])

        plt.tight_layout()
        plt.show()

    def plot_loss_history(self, history):
        plt.figure(figsize=(8, 5))
        plt.plot(history['train_loss'], label='Train Loss', marker='o')
        
        if 'val_loss' in history and history['val_loss']:
            plt.plot(history['val_loss'], label='Validation Loss', marker='x')

        plt.title('Training and Validation Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

