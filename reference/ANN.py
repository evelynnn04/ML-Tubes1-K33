'''
Ini aku copas dari seleksi asisten dulu yah, referensi yg aku pake buat ngerjain kemaren:
- github.com/lionelmessi6410/Neural-Networks-from-Scratch/blob/main/NN-from-Scratch.ipynb
- medium.com/@waleedmousa975/building-a-neural-network-from-scratch-using-numpy-and-math-libraries-a-step-by-step-tutorial-in-608090c20466

Laporan aku: 
https://docs.google.com/document/d/1IhYuueUO9kGcVBxmDwl363JiYDrB_B1w7dKZH5NyJB0/edit?usp=sharing

Semoga bermanfaat 🙏🙏
'''

import numpy as np
import time

class ANN_Selfmade():
    def __init__(self, sizes, activation='sigmoid'):
        self.sizes = sizes
    
        if activation == 'relu':
            self.activation = self.relu
        elif activation == 'sigmoid':
            self.activation = self.sigmoid
        else:
            raise ValueError("Activation function not supported, please use 'relu' or 'sigmoid' instead.")
    
        self.params = self.initialize()
        self.cache = {}
    
    def custom_rounding(arr, threshold=0.5):
        rounded_array = np.where(arr - np.floor(arr) > threshold, np.ceil(arr), np.floor(arr))
        return rounded_array

        
    def relu(self, x, derivative=False):
        if derivative:
            x = np.where(x < 0, 0, x)
            x = np.where(x >= 0, 1, x)
            return x
        return np.maximum(0, x)

    def sigmoid(self, x, derivative=False):
        x = np.clip(x, -1*2**8, 2**8)
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    def softmax(self, x):
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0)

    def initialize(self):
        input_layer=self.sizes[0]
        hidden_layer=self.sizes[1]
        output_layer=self.sizes[2]
        
        params = {
            "W1": np.random.randn(hidden_layer, input_layer) * np.sqrt(1./input_layer),
            "b1": np.zeros((hidden_layer, 1)) * np.sqrt(1./input_layer),
            "W2": np.random.randn(output_layer, hidden_layer) * np.sqrt(1./hidden_layer),
            "b2": np.zeros((output_layer, 1)) * np.sqrt(1./hidden_layer)
        }
        return params
    
    def initialize_momemtum_optimizer(self):
        momemtum_opt = {
            "W1": np.zeros(self.params["W1"].shape),
            "b1": np.zeros(self.params["b1"].shape),
            "W2": np.zeros(self.params["W2"].shape),
            "b2": np.zeros(self.params["b2"].shape),
        }
        return momemtum_opt

    def feed_forward(self, x):
        self.cache["X"] = x
        self.cache["Z1"] = np.dot(self.params["W1"], self.cache["X"].T) + self.params["b1"]
        self.cache["A1"] = self.activation(self.cache["Z1"])
        self.cache["Z2"] = np.dot(self.params["W2"], self.cache["A1"]) + self.params["b2"]
        self.cache["A2"] = self.sigmoid(self.cache["Z2"])
        return self.cache["A2"]
    
    def back_propagate(self, y, output):
        m = y.shape[0]

        Z1 = self.cache["Z1"]
        A1 = self.cache["A1"]
        Z2 = self.cache["Z2"]
        A2 = self.cache["A2"]
        
        epsilon = 1e-8
        dA2 = - (y/(A2 + epsilon)) + ((1-y)/(1-A2 + epsilon))
        dZ2 = np.clip(dA2 * (A2 * (1-A2)), -1, 1)
        
        dW2 = (1/m) * np.dot(dZ2, A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
        
        dA1 = np.dot(self.params["W2"].T, dZ2)
        dZ1 = dA1 * (A1 * (1-A1))

        dW1 = (1/m) * np.dot(dZ1, self.cache['X'])
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
        self.grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

        return self.grads 
    
    def cross_entropy_loss(self, y, output):
        epsilon = 1e-15
        output = np.clip(output, epsilon, 1 - epsilon)
        m = y.shape[0]

        l = -(1/m) * np.sum(y*np.log(output) + (1-y)*np.log(1-output))
        return l

                
    def optimize(self, l_rate=0.1, beta=.9):
        if self.optimizer == "sgd":
            for key in self.params:
                self.params[key] = self.params[key] - l_rate * self.grads[key]
        elif self.optimizer == "momentum":
            for key in self.params:
                self.momemtum_opt[key] = (beta * self.momemtum_opt[key] + (1. - beta) * self.grads[key])
                self.params[key] = self.params[key] - l_rate * self.momemtum_opt[key]
        else:
            raise ValueError("Optimizer not supported, please choose between 'sgd' or 'momentum'!")

    def accuracy(self, y, output):
        predictions = (output.T > 0.5).astype(int)
        return np.mean(predictions == y)


    def train(self, x_train, y_train, x_test, y_test, epochs=10, 
              batch_size=64, optimizer='momentum', l_rate=0.1, beta=.9):
        self.epochs = epochs
        self.batch_size = batch_size
        num_batches = -(-x_train.shape[0] // self.batch_size)
        '''
        Ini kaya pembagiannya gitu
        kenapa di - - ?
        Let say 103 // 32 = 3, padahal 32 * 3 = 96, jadi ada 7 row yg kelewat 
        Kalo - (-103 // 32) = - (-4) = 4 (actually it's around -3.2) 
        '''
        x_train = x_train.to_numpy()
        x_test = x_test.to_numpy()
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
        self.optimizer = optimizer
        if self.optimizer == 'momentum':
            self.momemtum_opt = self.initialize_momemtum_optimizer()
        
        start_time = time.time()
        template = "Epoch {}: {:.2f}s, train acc={:.2f}, train loss={:.2f}, test acc={:.2f}, test loss={:.2f}"
        
        for i in range(self.epochs):
            permutation = np.random.permutation(x_train.shape[0])
            x_train_shuffled = x_train[permutation]
            y_train_shuffled = y_train[permutation]

            for j in range(num_batches):
                begin = j * self.batch_size
                end = min(begin + self.batch_size, x_train.shape[0]-1)
                x = x_train_shuffled[begin:end]
                y = y_train_shuffled[begin:end]
                
                output = self.feed_forward(x)
                _ = self.back_propagate(y, output)
                self.optimize(l_rate=l_rate, beta=beta)

            output = self.feed_forward(x_train)
            output = np.where(output - np.floor(output) > 0.5, np.ceil(output), np.floor(output))
            train_acc = self.accuracy(y_train, output)
            train_loss = self.cross_entropy_loss(y_train, output)
            output = self.feed_forward(x_test)
            test_acc = self.accuracy(y_test, output)
            test_loss = self.cross_entropy_loss(y_test, output)
            print(template.format(i+1, time.time()-start_time, train_acc, train_loss, test_acc, test_loss))

    def predict(self, X):
        output = self.feed_forward(X)
        predictions = np.where(output.T > 0.5, 1, 0)
        return predictions