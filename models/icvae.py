from torch import flatten, exp, randn_like
import torch.nn as nn
import numpy as np


def reparameterize(mu, logvar):
    std = exp(0.5 * logvar)
    eps = randn_like(std)
    return mu + eps * std


def conv_shape(input_shape, kernel_size, padding, stride, pool_size=None, pool_stride=None):
    output_size = (np.array(input_shape) - kernel_size + 2 * padding) // stride + 1
    if pool_size:
        output_size = (output_size - pool_size) // pool_stride + 1

    return output_size


def conv_block(in_channels, out_channels, kernel_size, padding, stride, pool_size=None, pool_stride=None):
    layers = list()
    layers.append(nn.Conv3d(in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            stride=stride))
    layers.append(nn.BatchNorm3d(out_channels))
    if pool_size:
        layers.append(nn.MaxPool3d(kernel_size=pool_size, stride=pool_stride, return_indices=True))
    return nn.Sequential(*layers)


def tconv_block(in_channels, out_channels, kernel_size, padding, stride):
    layers = list()
    layers.append(nn.ConvTranspose3d(in_channels,
                                     out_channels,
                                     kernel_size=kernel_size,
                                     padding=padding,
                                     stride=stride))
    layers.append(nn.BatchNorm3d(out_channels))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """ Encoder based on the models from "Accurate brain age prediction with lightweight deep neural networks",
        Peng et al., 2020. Official repository: https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain/tree/master
    """
    def __init__(self,
                 input_shape=(160, 192, 160),
                 latent_dim=354,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 pool_size=2,
                 pool_stride=2,
                 last_kernel_size=1,
                 last_padding=0,
                 channels=(32, 64, 128, 256, 256, 64)):
        super(Encoder, self).__init__()
        self.channels = list(channels)
        self.n_blocks = len(channels)
        self.kernel_size, self.padding, self.stride = kernel_size, padding, stride
        self.pooling_kernel, self.pooling_stride = pool_size, pool_stride
        self.last_kernel, self.last_padding = last_kernel_size, last_padding

        self.conv_blocks = nn.ModuleList()
        for i in range(self.n_blocks):
            in_channels = 1 if i == 0 else self.channels[i - 1]
            if i < self.n_blocks - 1:
                self.conv_blocks.append(conv_block(in_channels, self.channels[i], kernel_size, padding, stride,
                                                   pool_size, pool_stride))
            else:
                self.conv_blocks.append(conv_block(in_channels, self.channels[i], last_kernel_size, last_padding, stride))
        self.activation = nn.ReLU()

        features_shape = self.features_shape(input_shape)
        self.fc_mu = nn.Linear(self.channels[-1] * np.prod(features_shape), latent_dim)
        self.fc_logvar = nn.Linear(self.channels[-1] * np.prod(features_shape), latent_dim)

    def forward(self, x):
        pooling_indices = []
        for i, block in enumerate(self.conv_blocks):
            if i < self.n_blocks - 1:
                x, indices = block(x)
                pooling_indices.append(indices)
            else:
                x = block(x)
            x = self.activation(x)
        x = flatten(x, 1)
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)

        return mu, logvar, pooling_indices

    def features_shape(self, input_shape):
        final_dim = np.array(input_shape)
        for i in range(self.n_blocks):
            if i < self.n_blocks - 1:
                final_dim = conv_shape(final_dim, self.kernel_size, self.padding, self.stride,
                                       self.pooling_kernel, self.pooling_stride)
            else:
                final_dim = conv_shape(final_dim, self.last_kernel, self.last_padding, self.stride)
        return final_dim


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


class ICVAE(nn.Module):
    def __init__(self,
                 input_shape=(160, 192, 160),
                 latent_dim=354,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 pool_size=2,
                 pool_stride=2,
                 last_kernel_size=1,
                 last_padding=0,
                 channels=(32, 64, 128, 256, 256, 64)):
        super(ICVAE, self).__init__()
        channels = list(channels)
        self.encoder = Encoder(input_shape, latent_dim, kernel_size, stride, padding, pool_size, pool_stride,
                               last_kernel_size, last_padding, channels)
        features_shape = self.encoder.features_shape(input_shape)
        channels.reverse()
        self.decoder = Decoder(features_shape, latent_dim, kernel_size, stride, padding, pool_size, pool_stride,
                               last_kernel_size, last_padding, channels)

    def forward(self, x):
        mu, logvar, pooling_indices = self.encoder(x)
        z = reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z, pooling_indices)
        return x_reconstructed, mu, logvar
