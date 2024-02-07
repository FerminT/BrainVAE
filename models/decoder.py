import numpy as np
from torch import nn as nn

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
                 channels=(64, 256, 256, 128, 64, 32)):
        super(Decoder, self).__init__()
        self.channels = list(channels)
        self.n_blocks = len(channels)
        self.input_shape = list(input_shape)

        self.fc_input = nn.Linear(latent_dim, self.channels[0] * np.prod(input_shape))
        self.tconv_blocks = nn.ModuleList()
        for i in range(self.n_blocks):
            out_channels = 1 if i == self.n_blocks - 1 else self.channels[i + 1]
            if i == 0:
                self.tconv_blocks.append(tconv_block(self.channels[i], out_channels, first_kernel_size, first_padding,
                                                     stride))
            else:
                self.tconv_blocks.append(tconv_block(self.channels[i], out_channels, kernel_size, padding, stride))
        self.activation = nn.ReLU()
        self.unpooling_layer = nn.MaxUnpool3d(kernel_size=unpool_size, stride=unpool_stride)

    def forward(self, x, pooling_indices):
        x = self.fc_input(x)
        x = x.view(-1, self.channels[0], *self.input_shape)
        for i, block in enumerate(self.tconv_blocks):
            if i > 0:
                x = self.unpooling_layer(x, indices=pooling_indices.pop())
            x = self.activation(block(x))
        return x
