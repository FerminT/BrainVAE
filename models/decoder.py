import numpy as np
from torch import nn as nn
from torch import cat
from models.utils import tconv_block


class Decoder(nn.Module):
    def __init__(self,
                 input_shape=(5, 6, 5),
                 latent_dim=354,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 first_kernel_size=1,
                 first_padding=0,
                 first_stride=1,
                 channels=(64, 256, 256, 128, 64, 32),
                 conditional_dim=0):
        super(Decoder, self).__init__()
        self.channels = list(channels)
        self.n_blocks = len(channels)
        self.input_shape = list(input_shape)
        self.conditional_dim = conditional_dim

        self.fc_input = nn.Linear(latent_dim + conditional_dim, self.channels[0] * np.prod(input_shape))
        self.tconv_blocks = build_modules(self.n_blocks, self.channels, kernel_size, stride, padding, first_kernel_size,
                                          first_padding, first_stride)

    def forward(self, x, condition):
        if self.conditional_dim > 0:
            if condition is None or condition.shape[-1] != self.conditional_dim:
                raise ValueError('Conditional dimension does not match the input dimension')
            x = cat([x, condition], dim=1)
        x = self.fc_input(x)
        x = x.view(-1, self.channels[0], *self.input_shape)
        x = self.tconv_blocks(x)
        return x


def build_modules(n_blocks, channels, kernel_size, stride, padding, first_kernel_size, first_padding, first_stride):
    modules = nn.ModuleList()
    for i in range(n_blocks):
        out_channels = 1 if i == n_blocks - 1 else channels[i + 1]
        if i == 0:
            modules.append(tconv_block(channels[i], out_channels, first_kernel_size, first_padding, first_stride))
        else:
            modules.append(tconv_block(channels[i], out_channels, kernel_size, padding, stride))
    return nn.Sequential(*modules)
