import numpy as np
from torch import nn as nn
from torch import cat
from models.utils import tconv_block


class Decoder(nn.Module):
    def __init__(self,
                 input_shape=(5, 6, 5),
                 latent_dim=354,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 unpool_size=2,
                 unpool_stride=2,
                 first_kernel_size=1,
                 first_padding=0,
                 channels=(64, 256, 256, 128, 64, 32),
                 conditional_dim=0):
        super(Decoder, self).__init__()
        self.channels = list(channels)
        self.n_blocks = len(channels)
        self.input_shape = list(input_shape)
        self.conditional_dim = conditional_dim

        self.fc_input = nn.Linear(latent_dim + conditional_dim, self.channels[0] * np.prod(input_shape))
        self.tconv_blocks = build_modules(self.n_blocks, self.channels, kernel_size, stride, padding, first_kernel_size,
                                          first_padding)
        self.activation = nn.ReLU()
        self.unpooling_layer = nn.MaxUnpool3d(kernel_size=unpool_size, stride=unpool_stride)

    def forward(self, x, pooling_indices, condition):
        if self.conditional_dim > 0:
            if condition is None or condition.shape[-1] != self.conditional_dim:
                raise ValueError('Conditional dimension does not match the input dimension')
            x = cat([x, condition], dim=1)
        x = self.fc_input(x)
        x = x.view(-1, self.channels[0], *self.input_shape)
        x = perform_deconvolution(x, pooling_indices, self.tconv_blocks, self.unpooling_layer, self.activation)
        return x


def perform_deconvolution(x, pooling_indices, tconv_blocks, unpooling_layer, activation):
    for i, block in enumerate(tconv_blocks):
        if i > 0:
            x = unpooling_layer(x, indices=pooling_indices.pop())
        x = activation(block(x))
    return x


def build_modules(n_blocks, channels, kernel_size, stride, padding, first_kernel_size, first_padding):
    modules = nn.ModuleList()
    for i in range(n_blocks):
        out_channels = 1 if i == n_blocks - 1 else channels[i + 1]
        if i == 0:
            modules.append(tconv_block(channels[i], out_channels, first_kernel_size, first_padding, stride))
        else:
            modules.append(tconv_block(channels[i], out_channels, kernel_size, padding, stride))
    return modules
