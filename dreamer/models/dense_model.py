import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DenseModel(nn.Module):
    def __init__(self, lenfeature , shape, layers, units, dist='normal', act=nn.ELU()):
        super(DenseModel, self).__init__()
        self._shape = shape
        self._layers = layers
        self._units = units
        self._dist = dist
        self._act = act
        # For adjusting pytorch to tensforflow
        self._lenfeature = lenfeature
        #Defining the structure of the NN
        self.fcfirst = nn.Linear(self._lenfeature , self._units )
        self.linears = nn.ModuleList([nn.Linear(self._units, self._units) for i in range(self._layers-1)])
        self.fclast = nn.Linear( self._units, int(np.prod(self._shape)))


    
    def __call__(self, features):
        x = features
        x = self._act(self.fcfirst(x))
        for l in self.linears:
            x = self._act(l(x))
        x = self.fclast(x)
        # TODO: Seems to do nothing in pytorch code. Come back if there are errors!
        x = torch.reshape(x, tuple(np.array(np.concatenate((features.shape[:-1], self._shape), axis=0), dtype=np.int)))     #It makes the size of output = (batch_size,)
        if self._dist == 'normal':
            return td.independent.Independent(td.Normal(x, 1), len(self._shape))
        if self._dist == 'binary':
            return td.independent.Independent(td.Bernoulli(logits=x), len(self._shape))
        raise NotImplementedError(self._dist)        
