import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn


class ObservationEncoder(nn.Module):
    def __init__(self, depth=32, stride=2, shape=(3, 64, 64), activation=nn.ReLU):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(shape[0], 1 * depth, 4, stride),
            activation(),
            nn.Conv2d(1 * depth, 2 * depth, 4, stride),
            activation(),
            nn.Conv2d(2 * depth, 4 * depth, 4, stride),
            activation(),
            nn.Conv2d(4 * depth, 8 * depth, 4, stride),
            activation(),
        )

    def forward(self, obs):
        batch_shape = obs.shape[:-3]
        img_shape = obs.shape[-3:]
        embed = self.convolutions(obs.reshape(-1, *img_shape))
        embed = torch.reshape(embed, (*batch_shape, -1))
        return embed


class ObservationDecoder(nn.Module):
    def __init__(self, depth=32, stride=2, activation=nn.ReLU, embed_size=1024, shape=(3, 64, 64),
                 distribution=td.Normal):
        super().__init__()
        self.linear = nn.Linear(embed_size, 32 * depth)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32 * depth, 4 * depth, 5, stride),
            activation(),
            nn.ConvTranspose2d(4 * depth, 2 * depth, 5, stride),
            activation(),
            nn.ConvTranspose2d(2 * depth, 1 * depth, 6, stride),
            activation(),
            nn.ConvTranspose2d(1 * depth, shape[0], 6, stride),
        )
        self.depth = depth
        self.shape = shape
        self.distribution = distribution

    def forward(self, x):
        """
        :param x: size(*batch_shape, embed_size)
        :return: obs_dist = size(*batch_shape, *self.shape)
        """
        batch_shape = x.shape[:-1]
        embed_size = x.shape[-1]
        squeezed_size = np.prod(batch_shape).item()
        x = x.reshape(squeezed_size, embed_size)
        x = self.linear(x)
        x = torch.reshape(x, (squeezed_size, 32 * self.depth, 1, 1))
        x = self.decoder(x)
        mean = torch.reshape(x, (*batch_shape, *self.shape))
        obs_dist = self.distribution(mean, 1)
        return obs_dist
