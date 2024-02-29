import numpy as np
from torch import exp, randn_like, nn as nn


def reparameterize(mu, logvar):
    std = exp(0.5 * logvar)
    eps = randn_like(std)
    return mu + eps * std


def conv_shape(input_shape, kernel_size, padding, stride, pool_size=0, pool_stride=0):
    output_size = (np.array(input_shape) - kernel_size + 2 * padding) // stride + 1
    if pool_size > 0:
        output_size = (output_size - pool_size) // pool_stride + 1
    return output_size


def conv_block(in_channels, out_channels, kernel_size, padding, stride, pool_size=0, pool_stride=0):
    layers = list()
    layers.append(nn.Conv3d(in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            stride=stride))
    layers.append(nn.BatchNorm3d(out_channels))
    if pool_size > 0:
        layers.append(nn.MaxPool3d(kernel_size=pool_size, stride=pool_stride))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def tconv_block(in_channels, out_channels, kernel_size, padding, stride):
    layers = list()
    layers.append(nn.ConvTranspose3d(in_channels,
                                     out_channels,
                                     kernel_size=kernel_size,
                                     padding=padding,
                                     stride=stride,
                                     output_padding=padding))
    layers.append(nn.BatchNorm3d(out_channels))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)
