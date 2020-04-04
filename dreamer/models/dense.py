import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn


class DenseModel(nn.Module):
    def __init__(self, feature_size, output_shape, layers, hidden_size, dist='normal', activation=nn.ELU):
        super().__init__()
        self._output_shape = output_shape
        self._layers = layers
        self._hidden_size = hidden_size
        self._dist = dist
        self.activation = activation
        # For adjusting pytorch to tensorflow
        self._feature_size = feature_size
        # Defining the structure of the NN
        self.model = self.build_model()

    def build_model(self):
        model = [nn.Linear(self._feature_size, self._hidden_size)]
        model += [self.activation()]
        for i in range(self._layers - 1):
            model += [nn.Linear(self._hidden_size, self._hidden_size)]
            model += [self.activation()]
        model += [nn.Linear(self._hidden_size, int(np.prod(self._output_shape)))]
        return nn.Sequential(*model)

    def forward(self, features):
        x = self.model(features)
        # TODO: Seems to do nothing in pytorch code. Come back if there are errors!
        x = torch.reshape(x, tuple(np.array(np.concatenate((features.shape[:-1], self._output_shape), axis=0),
                                            dtype=np.int)))  # It makes the size of output = (batch_size,)
        if self._dist == 'normal':
            return td.independent.Independent(td.Normal(x, 1), len(self._output_shape))
        if self._dist == 'binary':
            return td.independent.Independent(td.Bernoulli(logits=x), len(self._output_shape))
        raise NotImplementedError(self._dist)
