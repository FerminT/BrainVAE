import numpy as np
from torch import optim, exp, randn_like, nn as nn


def reparameterize(mu, logvar):
    std = exp(0.5 * logvar)
    eps = randn_like(std)
    return mu + eps * std


def features_shape(input_shape, layers):
    final_dim = np.array(input_shape)
    for layer in layers:
        layer_cfg = layers[layer]
        final_dim = conv_shape(final_dim, layer_cfg['kernel_size'], layer_cfg['padding'], layer_cfg['stride'],
                               layer_cfg['pool_size'], layer_cfg['pool_stride'])
    return final_dim


def conv_shape(input_shape, kernel_size, padding, stride, pool_size=0, pool_stride=0):
    output_size = (np.array(input_shape) - kernel_size + 2 * padding) // stride + 1
    if pool_size > 0:
        output_size = (output_size - pool_size) // pool_stride + 1
    return output_size


def conv_block(in_channels, out_channels, kernel_size, padding, stride, pool_size=0, pool_stride=0, batch_norm=True,
               activation='relu'):
    layers = list()
    layers.append(nn.Conv3d(in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            stride=stride))
    if batch_norm:
        layers.append(nn.BatchNorm3d(out_channels))
    if pool_size > 0:
        layers.append(nn.MaxPool3d(kernel_size=pool_size, stride=pool_stride))
    add_activation_layer(layers, activation)
    return nn.Sequential(*layers)


def tconv_block(in_channels, out_channels, kernel_size, padding, stride, batch_norm=True, activation='relu'):
    layers = list()
    layers.append(nn.ConvTranspose3d(in_channels,
                                     out_channels,
                                     kernel_size=kernel_size,
                                     padding=padding,
                                     stride=stride,
                                     output_padding=padding))
    if batch_norm:
        layers.append(nn.BatchNorm3d(out_channels))
    add_activation_layer(layers, activation)
    return nn.Sequential(*layers)


def add_activation_layer(layers, activation):
    if activation == 'relu':
        layers.append(nn.ReLU())
    elif activation == 'leakyrelu':
        layers.append(nn.LeakyReLU())
    elif activation == 'tanh':
        layers.append(nn.Tanh())
    elif activation == 'sigmoid':
        layers.append(nn.Sigmoid())


def init_optimizer(optimizer, parameters, lr, momentum, weight_decay):
    if optimizer == 'AdamW':
        optimizer = optim.AdamW(parameters, lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay)
    elif optimizer == 'SGD':
        optimizer = optim.SGD(parameters, lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    else:
        optimizer = getattr(optim, optimizer)(parameters, lr=lr)
    return optimizer


def get_latent_representation(t1_img, encoder):
    mu, logvar = encoder(t1_img)
    z = reparameterize(mu, logvar)
    return z
