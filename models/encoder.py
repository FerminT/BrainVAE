from torch import nn as nn, flatten
from models.utils import conv_block, features_shape
import numpy as np


class Encoder(nn.Module):
    """ Encoder based on the models from "Accurate brain age prediction with lightweight deep neural networks",
        Peng et al., 2020. Official repository: https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain/tree/master
    """
    def __init__(self,
                 input_shape,
                 latent_dim,
                 blocks):
        super(Encoder, self).__init__()
        self.conv_blocks = nn.ModuleList()
        prev_channels = 1
        for block in blocks:
            conv_layer = blocks[block]
            self.conv_blocks.append(conv_block(in_channels=prev_channels,
                                               out_channels=conv_layer['channels'],
                                               kernel_size=conv_layer['kernel_size'],
                                               padding=conv_layer['padding'],
                                               stride=conv_layer['stride'],
                                               pool_size=conv_layer['pool_size'],
                                               pool_stride=conv_layer['pool_stride'],
                                               batch_norm=conv_layer['batch_norm'],
                                               activation=conv_layer['activation']))
            prev_channels = conv_layer['channels']
        self.conv_blocks = nn.Sequential(*self.conv_blocks)

        self.final_shape = features_shape(input_shape, blocks)
        self.final_shape = np.insert(self.final_shape, 0, prev_channels)
        self.fc_mu = nn.Linear(np.prod(self.final_shape), latent_dim)
        self.fc_logvar = nn.Linear(np.prod(self.final_shape), latent_dim)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = flatten(x, 1)
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)

        return mu, logvar
