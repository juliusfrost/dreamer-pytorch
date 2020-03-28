import torch
import torch.distributions as td
import torch.nn as nn


class ObservationEncoder(nn.Module):
    def __init__(self, depth=32, stride=2, activation=nn.ReLU):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(3, 1 * depth, 4, stride),
            activation(),
            nn.Conv2d(1 * depth, 2 * depth, 4, stride),
            activation(),
            nn.Conv2d(2 * depth, 4 * depth, 4, stride),
            activation(),
            nn.Conv2d(4 * depth, 8 * depth, 4, stride),
            activation(),
        )

    def forward(self, obs):
        embed = self.convolutions(obs)
        return torch.reshape(embed, (embed.size(0), -1))


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
        x = self.linear(x)
        x = torch.reshape(x, (-1, 32 * self.depth, 1, 1))
        x = self.decoder(x)
        mean = torch.reshape(x, (x.size(0), *self.shape))
        return self.distribution(mean, 1)
