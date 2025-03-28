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

class Layer:
    def __init__(self,n_neurons=3, init='zero', activation='relu', weights=None, biases=None):
        # weights & biases ditambah di init buat memfasilitasi loading weight & bias
        self.n_neurons = n_neurons
        self.current_input_batch = None
        self.init = init    # zero/uniform/normal/xavier/he - harusnya gaperlu disini, ini di layer langsung harusnya
        self.weights = weights
        self.biases = biases
        self.activation = activation    # linear/relu/sigmoid/tanh/softmax/binary_step/leaky_relu/prelu/elu/swish(ini)/gelu(ini)
        self.learning_rate = None

        self.grad_weights = None
        self.grad_biases = None 

        self.net = None
        self.out = None

    def __update_weights_dEdW(self, dEdW):
        clip_value = 1.0  # maksimal update yang diperbolehkan
        dEdW = np.clip(dEdW, -clip_value, clip_value)
        self.weights -= self.learning_rate * dEdW

    def __update_biases_dEdB(self, dEdB):
        clip_value = 1.0  # maksimal update yang diperbolehkan
        dEdB = np.clip(dEdB, -clip_value, clip_value)
        self.biases -= self.learning_rate * dEdB

    def __update_weights_err_term(self, err_term):
        for input in self.current_input_batch:
            for i in self.weights:
                for j in i:
                    self.weights += self.learning_rate*err_term[j]*self.input[i]

    def __update_biases_err_term(self, err_term):
        for i in self.weights:
            for j in i:
                self.weights += self.learning_rate*err_term[j]*1

    def forward(self, current_input_batch):
        self.current_input_batch = current_input_batch

        if(self.weights is None):
            
            if(self.init == 'zero'):
                self.weights = np.array([[VarValue(0,varname='w_'+str(uuid.uuid4())) for _ in range(self.n_neurons)] for _ in range(len(self.current_input_batch[0]))])
                self.biases = np.array([VarValue(0,varname='b_'+str(uuid.uuid4())) for _ in range(self.n_neurons)])
                print("TESTT")
                print(self.weights)
                print(self.biases)

            # Ini semua diround soalnya hasil operasinya kegedean, kena warning wkwkwk. CMIIW yh harusnya berapa angka di belakang koma - @evelynnn04

            elif(self.init == 'uniform'):
                limit = 1 / np.sqrt(len(self.current_input_batch[0]))
                self.weights = np.array([[VarValue(round(np.random.uniform(-limit, limit), 4), varname='w_'+str(uuid.uuid4())) 
                                        for _ in range(self.n_neurons)] for _ in range(len(self.current_input_batch[0]))])
                self.biases = np.array([VarValue(round(np.random.uniform(-limit, limit), 4), varname='b_'+str(uuid.uuid4())) 
                                        for _ in range(self.n_neurons)])

            elif self.init == 'normal':
                std = 1 / np.sqrt(len(self.current_input_batch[0]))
                self.weights = np.array([[VarValue(round(np.random.normal(0, std), 4), varname='w_'+str(uuid.uuid4())) 
                                        for _ in range(self.n_neurons)] for _ in range(len(self.current_input_batch[0]))])
                self.biases = np.array([VarValue(round(np.random.normal(0, std), 4), varname='b_'+str(uuid.uuid4())) 
                                        for _ in range(self.n_neurons)])

            elif self.init == 'xavier':
                std = np.sqrt(2 / (len(self.current_input_batch[0]) + self.n_neurons))
                self.weights = np.array([[VarValue(round(np.random.normal(0, std), 4), varname='w_'+str(uuid.uuid4())) 
                                        for _ in range(self.n_neurons)] for _ in range(len(self.current_input_batch[0]))])
                self.biases = np.array([VarValue(round(0, 4), varname='b_'+str(uuid.uuid4())) 
                                        for _ in range(self.n_neurons)])

            elif self.init == 'he':
                std = np.sqrt(2 / len(self.current_input_batch[0]))
                self.weights = np.array([[VarValue(round(np.random.normal(0, std), 4), varname='w_'+str(uuid.uuid4())) 
                                        for _ in range(self.n_neurons)] for _ in range(len(self.current_input_batch[0]))])
                self.biases = np.array([VarValue(round(0, 4), varname='b_'+str(uuid.uuid4())) 
                                        for _ in range(self.n_neurons)])


        self.net = (np.dot(self.current_input_batch, self.weights)) + self.biases

        if(self.activation == 'linear'):
            self.out = self.net

        elif(self.activation == 'relu'):
            self.out = np.array([[net.relu() for net in row] for row in self.net])

        elif(self.activation == 'sigmoid'):
            for i in self.net:
                for j in i:
                    j.value = np.clip(j.value, -500, 500)
            self.out = 1 / (1 + (math.exp(1))**(-self.net))

        elif(self.activation == 'tanh'):
            self.out = (math.exp(1)**self.net - math.exp(1)**(-self.net))/(math.exp(1)**self.net + math.exp(1)**(-self.net))

        elif self.activation == 'softmax':
            exp_values = np.array([[net.exp() for net in row] for row in self.net])
            sums = np.array([[VarValue(sum(n.value for n in row), varname='sum_exp_'+str(uuid.uuid4()))] for row in exp_values])
            self.out = np.array([[exp_values[i][j] / sums[i][0] for j in range(len(exp_values[i]))] for i in range(len(exp_values))])


    def backward(self, err=None):
        if err is None:
            raise ValueError("err (VarValue object) harus diberikan dari fungsi loss sebagai parameter!")

        # Ambil derivatif langsung dari err (loss)
        dEdW = np.zeros((len(self.weights), len(self.weights[0])))
        dEdB = np.zeros(len(self.biases))

        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                varname = self.weights[i][j].varname
                dEdW[i][j] = err.derivative_to.get(varname, 0.0)

        for j in range(len(self.biases)):
            varname = self.biases[j].varname
            dEdB[j] = err.derivative_to.get(varname, 0.0)

        # Simpan gradien
        self.grad_weights = dEdW
        self.grad_biases = dEdB

        # Update bobot dan bias
        clip_val = 1.0
        self.weights -= self.learning_rate * np.clip(dEdW, -clip_val, clip_val)
        self.biases -= self.learning_rate * np.clip(dEdB, -clip_val, clip_val)


    def clean_derivative(self):
        for input in self.current_input_batch:
            for x in input:
                x.derivative_to.clear()
                x.children = ()

        for i in self.weights:
            for j in i:
                j.derivative_to.clear()
                j.children = ()

        for b in self.biases:
            b.derivative_to ={}
            b.children = ()

        for i in self.net:
            for j in i:
                j.derivative_to.clear()
                j.children = ()

        for i in self.out:
            for j in i:
                j.derivative_to.clear()
                j.children = ()