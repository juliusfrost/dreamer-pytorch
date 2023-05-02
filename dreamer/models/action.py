import numpy as np
import torch
import torch.distributions
import torch.nn as nn
import torch.nn.functional as F

from dreamer.models.distribution import SampleDist
from torch.distributions import TanhTransform as TanhBijector


class ActionDecoder(nn.Module):
    def __init__(
        self,
        action_size,
        feature_size,
        hidden_size,
        layers,
        dist="tanh_normal",
        activation=nn.ELU,
        min_std=1e-4,
        init_std=5,
        mean_scale=5,
    ):
        super().__init__()
        self.action_size = action_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.dist = dist
        self.activation = activation
        self.min_std = min_std
        self.init_std = init_std
        self.mean_scale = mean_scale
        self.feedforward_model = self.build_model()
        self.raw_init_std = np.log(np.exp(self.init_std) - 1)

    def build_model(self):
        model = [nn.Linear(self.feature_size, self.hidden_size)]
        model += [self.activation()]
        for i in range(1, self.layers):
            model += [nn.Linear(self.hidden_size, self.hidden_size)]
            model += [self.activation()]
        if self.dist == "tanh_normal":
            model += [nn.Linear(self.hidden_size, self.action_size * 2)]
        elif self.dist == "one_hot" or self.dist == "relaxed_one_hot":
            model += [nn.Linear(self.hidden_size, self.action_size)]
        else:
            raise NotImplementedError(f"{self.dist} not implemented")
        return nn.Sequential(*model)

    def forward(self, state_features):
        x = self.feedforward_model(state_features)
        dist = None
        if self.dist == "tanh_normal":
            mean, std = torch.chunk(x, 2, -1)
            mean = self.mean_scale * torch.tanh(mean / self.mean_scale)
            std = F.softplus(std + self.raw_init_std) + self.min_std
            dist = torch.distributions.Normal(mean, std)
            dist = torch.distributions.TransformedDistribution(dist, TanhBijector())
            dist = torch.distributions.Independent(dist, 1)
            dist = SampleDist(dist)
        elif self.dist == "one_hot":
            dist = torch.distributions.OneHotCategorical(logits=x)
        elif self.dist == "relaxed_one_hot":
            dist = torch.distributions.RelaxedOneHotCategorical(0.1, logits=x)
        return dist
