import numpy as np
from torch import nn as nn
from torch import cat, relu
from models.utils import tconv_block


class Decoder(nn.Module):
    def __init__(self,
                 input_shape,
                 latent_dim,
                 blocks,
                 conditional_dim=0):
        super(Decoder, self).__init__()
        self.input_shape = list(input_shape)
        self.conditional_dim = conditional_dim

        self.fc_input = nn.Linear(latent_dim + conditional_dim, np.prod(input_shape))
        self.tconv_blocks = build_modules(blocks)

    def forward(self, x, condition):
        if condition is not None:
            if 0 < self.conditional_dim != condition.shape[-1]:
                raise ValueError('Conditional dimension does not match the input dimension')
            x = cat([x, condition], dim=1) if self.conditional_dim > 0 else x + condition
        x = relu(self.fc_input(x))
        x = x.view(-1, *self.input_shape)
        x = self.tconv_blocks(x)
        return x


def build_modules(blocks):
    modules = nn.ModuleList()
    channels = [blocks[layer]['channels'] for layer in blocks]
    for i, block in enumerate(blocks):
        tconv_layer = blocks[block]
        if i == len(blocks) - 1:
            batch_norm, activation = None, None
            out_channels = 1
        else:
            batch_norm, activation = tconv_layer['batch_norm'], tconv_layer['activation']
            out_channels = channels[i + 1]
        modules.append(tconv_block(in_channels=tconv_layer['channels'],
                                   out_channels=out_channels,
                                   kernel_size=tconv_layer['kernel_size'],
                                   padding=tconv_layer['padding'],
                                   stride=tconv_layer['stride'],
                                   bias=tconv_layer['bias'],
                                   batch_norm=batch_norm,
                                   activation=activation))
    return nn.Sequential(*modules)
