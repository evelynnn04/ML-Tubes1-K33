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

class VarValue:
    def __init__(self, value, children=(), varname=''):
        self.varname = varname
        self.value = value
        self.derivative_to = {}
        self.children = children

    def __chain_rule(self, dSelfdx, child):
        if(child.varname[:5] != 'const'):
            for grandchild_varname in child.derivative_to:
                if(grandchild_varname != 'const'):
                    if(grandchild_varname in self.derivative_to):

                        self.derivative_to[grandchild_varname] += dSelfdx * child.derivative_to[grandchild_varname]
                    else:
                        self.derivative_to[grandchild_varname] = dSelfdx * child.derivative_to[grandchild_varname]
            if(len(self.derivative_to) == 0):
              raise ValueError(self.varname, child.varname)

    def relu(self):
        out = VarValue(max(0,self.value), children=(self,), varname='out_relu_'+str(uuid.uuid4()))
        if(self.varname[:5] != 'const'):
            dodx = 0 if self.value <= 0 else 1
            if(len(self.children) == 0):
                out.derivative_to[self.varname] = dodx
            else:
                out.__chain_rule(dodx, self)
        return out

    def ln(self):
        out = VarValue(math.log(self.value), children=(self,), varname='out_ln_'+str(uuid.uuid4()))
        if(self.varname[:5] != 'const'):
            dodx = 1/(self.value)
            if(len(self.children) == 0):
                out.derivative_to[self.varname] = dodx
            else:
                out.__chain_rule(dodx, self)
        return out

    def __mul__(self, other):
        other = other if isinstance(other, VarValue) else VarValue(other, varname='const'+str(uuid.uuid4()))
        out = VarValue(self.value * other.value, children=(self, other), varname='out_mul_'+str(uuid.uuid4()))

        if(self.varname[:5] != 'const'):
            dodx1 = other.value
            if(len(self.children) == 0):
                out.derivative_to[self.varname] = dodx1
            else:
                out.__chain_rule(dodx1, self)
        if(other.varname[:5] != 'const'):
            dodx2 = self.value
            if(len(other.children) == 0):
                out.derivative_to[other.varname] = dodx2
            else:
                out.__chain_rule(dodx2, other)
        return out

    def __add__(self, other):
        other = other if isinstance(other, VarValue) else VarValue(other, varname='const'+str(uuid.uuid4()))
        out = VarValue(self.value + other.value, children=(self, other), varname='out_add_'+str(uuid.uuid4()))
        if(self.varname[:5] != 'const'):
            dodx1 = 1
            if(len(self.children) == 0):
                out.derivative_to[self.varname] = dodx1
            else:
                out.__chain_rule(dodx1, self)
        if(other.varname[:5] != 'const'):
            dodx2 = 1
            if(len(other.children) == 0):
                out.derivative_to[other.varname] = dodx2
            else:
                out.__chain_rule(dodx2, other)
        return out

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other**-1

    def __neg__(self):
        return self * -1

    def __pow__(self, other):
        other = other if isinstance(other, VarValue) else VarValue(other, varname='const'+str(uuid.uuid4()))
        try:
            with errstate(over='raise', invalid='raise'):
                result = float(self.value) ** float(other.value)
        except (OverflowError, FloatingPointError):
            if abs(float(self.value)) > 1:
                result = math.inf
            else:
                result = 0

        out = VarValue(result, children=(self, other), varname='out_pow_'+str(uuid.uuid4()))

        if(self.varname[:5] != 'const'):
            dodx = other.value * self.value**(other.value-1)
            if(len(self.children) == 0):
                out.derivative_to[self.varname] = dodx
            else:
                out.__chain_rule(dodx, self)
        return out

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return other + (-self)

    def __rpow__(self, other):
        other = other if isinstance(other, VarValue) else VarValue(other, varname='const'+str(uuid.uuid4()))
        out = VarValue(other.value**self.value, children=(self, other), varname='out_rpow_'+str(uuid.uuid4()))
        if self.varname:
            dodx = other.value**self.value * math.log(other.value)
            if(len(self.children) == 0):
                out.derivative_to[self.varname] = dodx
            else:
                out.__chain_rule(dodx, self)
        return out

    def __rtruediv__(self, other):
        return other * self**-1

    # Equality
    def __eq__(self, other):
        if isinstance(other, VarValue):
            return self.varname == other.varname
        return self.varname == other

    # Inequality
    def __ne__(self, other):
        return not self.__eq__(other)

    # Less than
    def __lt__(self, other):
        if isinstance(other, VarValue):
            return self.value < other.value
        return self.value < other

    # Less than or equal
    def __le__(self, other):
        if isinstance(other, VarValue):
            return self.value <= other.value
        return self.value <= other

    # Greater than
    def __gt__(self, other):
        if isinstance(other, VarValue):
            return self.value > other.value
        return self.value > other

    # Greater than or equal
    def __ge__(self, other):
        if isinstance(other, VarValue):
            return self.value >= other.value
        return self.value >= other
    
    def log(self):
        out = VarValue(math.log(self.value), children=(self,), varname='outlog'+str(uuid.uuid4()))
        if self.varname[:5] != 'const':
            dodx = 1 / self.value
            if len(self.children) == 0:
                out.derivative_to[self.varname] = dodx
            else:
                out.__chain_rule(dodx, self)
        return out

    def exp(self):
        out = VarValue(math.exp(self.value), children=(self,), varname='out_exp_'+str(uuid.uuid4()))
        if self.varname[:5] != 'const':
            dodx = math.exp(self.value)
            if len(self.children) == 0:
                out.derivative_to[self.varname] = dodx
            else:
                out.__chain_rule(dodx, self)
        return out