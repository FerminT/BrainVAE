from torch import nn as nn, flatten
from models.utils import conv_block, conv_shape
import numpy as np


class Encoder(nn.Module):
    """ Encoder based on the models from "Accurate brain age prediction with lightweight deep neural networks",
        Peng et al., 2020. Official repository: https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain/tree/master
    """
    def __init__(self,
                 input_shape=(160, 192, 160),
                 latent_dim=354,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 pool_size=0,
                 pool_stride=0,
                 last_kernel_size=1,
                 last_padding=0,
                 last_stride=1,
                 channels=(32, 64, 128, 256, 256, 64)):
        super(Encoder, self).__init__()
        self.channels = list(channels)
        self.n_blocks = len(channels)
        self.kernel_size, self.padding, self.stride = kernel_size, padding, stride
        self.pooling_kernel, self.pooling_stride = pool_size, pool_stride
        self.last_kernel, self.last_padding, self.last_stride = last_kernel_size, last_padding, last_stride

        self.conv_blocks = nn.ModuleList()
        for i in range(self.n_blocks):
            in_channels = 1 if i == 0 else self.channels[i - 1]
            if i < self.n_blocks - 1:
                self.conv_blocks.append(conv_block(in_channels, self.channels[i], kernel_size, padding, stride,
                                                   pool_size, pool_stride))
            else:
                self.conv_blocks.append(
                    conv_block(in_channels, self.channels[i], last_kernel_size, last_padding, last_stride))
        self.conv_blocks = nn.Sequential(*self.conv_blocks)

        features_shape = self.features_shape(input_shape)
        self.fc_mu = nn.Linear(self.channels[-1] * np.prod(features_shape), latent_dim)
        self.fc_logvar = nn.Linear(self.channels[-1] * np.prod(features_shape), latent_dim)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = flatten(x, 1)
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)

        return mu, logvar

    def features_shape(self, input_shape):
        final_dim = np.array(input_shape)
        for i in range(self.n_blocks):
            if i < self.n_blocks - 1:
                final_dim = conv_shape(final_dim, self.kernel_size, self.padding, self.stride,
                                       self.pooling_kernel, self.pooling_stride)
            else:
                final_dim = conv_shape(final_dim, self.last_kernel, self.last_padding, self.last_stride)
        return final_dim
