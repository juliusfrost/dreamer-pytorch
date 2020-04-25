import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn


class DenseModel(nn.Module):
    def __init__(self, feature_size: int, output_shape: tuple, layers: int, hidden_size: int, dist='normal',
                 activation=nn.ELU):
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
        dist_inputs = self.model(features)
        reshaped_inputs = torch.reshape(dist_inputs, features.shape[:-1] + self._output_shape)
        if self._dist == 'normal':
            return td.independent.Independent(td.Normal(reshaped_inputs, 1), len(self._output_shape))
        if self._dist == 'binary':
            return td.independent.Independent(td.Bernoulli(logits=reshaped_inputs), len(self._output_shape))
        raise NotImplementedError(self._dist)
