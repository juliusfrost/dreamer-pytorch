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
        self.shape = shape
        self.stride = stride
        self.depth = depth

    def forward(self, obs):
        batch_shape = obs.shape[:-3]
        img_shape = obs.shape[-3:]
        embed = self.convolutions(obs.reshape(-1, *img_shape))
        embed = torch.reshape(embed, (*batch_shape, -1))
        return embed

    @property
    def embed_size(self):
        conv1_shape = conv_out_shape(self.shape[1:], 0, 4, self.stride)
        conv2_shape = conv_out_shape(conv1_shape, 0, 4, self.stride)
        conv3_shape = conv_out_shape(conv2_shape, 0, 4, self.stride)
        conv4_shape = conv_out_shape(conv3_shape, 0, 4, self.stride)
        embed_size = 8 * self.depth * np.prod(conv4_shape).item()
        return embed_size


class ObservationDecoder(nn.Module):
    def __init__(self, depth=32, stride=2, activation=nn.ReLU, embed_size=1024, shape=(3, 64, 64)):
        super().__init__()
        self.depth = depth
        self.shape = shape

        c, h, w = shape
        conv1_kernel_size = 6
        conv2_kernel_size = 6
        conv3_kernel_size = 5
        conv4_kernel_size = 5
        padding = 0
        conv1_shape = conv_out_shape((h, w), padding, conv1_kernel_size, stride)
        conv1_pad = output_padding_shape((h, w), conv1_shape, padding, conv1_kernel_size, stride)
        conv2_shape = conv_out_shape(conv1_shape, padding, conv2_kernel_size, stride)
        conv2_pad = output_padding_shape(conv1_shape, conv2_shape, padding, conv2_kernel_size, stride)
        conv3_shape = conv_out_shape(conv2_shape, padding, conv3_kernel_size, stride)
        conv3_pad = output_padding_shape(conv2_shape, conv3_shape, padding, conv3_kernel_size, stride)
        conv4_shape = conv_out_shape(conv3_shape, padding, conv4_kernel_size, stride)
        conv4_pad = output_padding_shape(conv3_shape, conv4_shape, padding, conv4_kernel_size, stride)
        self.conv_shape = (32 * depth, *conv4_shape)
        self.linear = nn.Linear(embed_size, 32 * depth * np.prod(conv4_shape).item())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32 * depth, 4 * depth, conv4_kernel_size, stride, output_padding=conv4_pad),
            activation(),
            nn.ConvTranspose2d(4 * depth, 2 * depth, conv3_kernel_size, stride, output_padding=conv3_pad),
            activation(),
            nn.ConvTranspose2d(2 * depth, 1 * depth, conv2_kernel_size, stride, output_padding=conv2_pad),
            activation(),
            nn.ConvTranspose2d(1 * depth, shape[0], conv1_kernel_size, stride, output_padding=conv1_pad),
        )

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
        x = torch.reshape(x, (squeezed_size, *self.conv_shape))
        x = self.decoder(x)
        mean = torch.reshape(x, (*batch_shape, *self.shape))
        obs_dist = td.Independent(td.Normal(mean, 1), len(self.shape))
        return obs_dist


def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2. * padding - (kernel_size - 1.) - 1.) / stride + 1.)


def output_padding(h_in, conv_out, padding, kernel_size, stride):
    return h_in - (conv_out - 1) * stride + 2 * padding - (kernel_size - 1) - 1


def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)


def output_padding_shape(h_in, conv_out, padding, kernel_size, stride):
    return tuple(output_padding(h_in[i], conv_out[i], padding, kernel_size, stride) for i in range(len(h_in)))
