import numpy as np
from scipy.stats import norm
from torch import optim, exp, randn_like, nn as nn
from scripts.constants import BRAIN_MASK


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


def conv_block(in_channels, out_channels, kernel_size, padding, stride, pool_size=0, pool_stride=0,
               bias=True, batch_norm=True, activation='relu'):
    layers = list()
    layers.append(nn.Conv3d(in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            stride=stride,
                            bias=bias))
    if batch_norm:
        layers.append(nn.BatchNorm3d(out_channels))
    if pool_size > 0:
        layers.append(nn.MaxPool3d(kernel_size=pool_size, stride=pool_stride))
    add_activation_layer(layers, activation)
    return nn.Sequential(*layers)


def tconv_block(in_channels, out_channels, kernel_size, padding, stride, bias=True, batch_norm=True, activation='relu'):
    layers = list()
    layers.append(nn.ConvTranspose3d(in_channels,
                                     out_channels,
                                     kernel_size=kernel_size,
                                     padding=padding,
                                     stride=stride,
                                     output_padding=padding,
                                     bias=bias))
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
        optimizer = optim.AdamW(parameters, lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay, eps=1e-4)
    elif optimizer == 'SGD':
        optimizer = optim.SGD(parameters, lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    else:
        optimizer = getattr(optim, optimizer)(parameters, lr=lr)
    return optimizer


def get_latent_representation(t1_img, encoder):
    mu, logvar = encoder(t1_img)
    z = reparameterize(mu, logvar)
    return z


def crop_center(data, shape):
    x, y, z = data.shape[-3:]
    start_x = (x - shape[0]) // 2
    start_y = (y - shape[1]) // 2
    start_z = (z - shape[2]) // 2
    cropped = data[..., start_x:start_x + shape[0], start_y:start_y + shape[1], start_z:start_z + shape[2]]
    return cropped


def crop_brain(img_batch):
    return crop_center(img_batch, BRAIN_MASK)


def num2vect(x, bin_range, bin_step=1, sigma=1):
    """
    bin_range: (start, end), size-2 tuple
    bin_step: should be a divisor of |end-start|
    sigma:
    = 0 for 'hard label', in which case v is an index
    > 0 for 'soft label', in which case v is a vector
    """
    bin_start = bin_range[0]
    bin_end = bin_range[1]
    bin_length = bin_end - bin_start
    if not bin_length % bin_step == 0:
        raise ValueError("bin's range should be divisible by bin_step!")
    bin_number = int(bin_length / bin_step)
    bin_centers = bin_start + float(bin_step) / 2 + bin_step * np.arange(bin_number)

    if sigma == 0:
        x = np.array(x, dtype=np.float32)
        i = np.floor((x - bin_start) / bin_step)
        i = i.astype(int)
        return i, bin_centers
    elif sigma > 0:
        if np.isscalar(x):
            v = np.zeros((bin_number,), dtype=np.float32)
            for i in range(bin_number):
                x1 = bin_centers[i] - float(bin_step) / 2
                x2 = bin_centers[i] + float(bin_step) / 2
                cdfs = norm.cdf([x1, x2], loc=x, scale=sigma)
                v[i] = cdfs[1] - cdfs[0]
            return v, bin_centers
        else:
            v = np.zeros((len(x), bin_number), dtype=np.float32)
            for j in range(len(x)):
                for i in range(bin_number):
                    x1 = bin_centers[i] - float(bin_step) / 2
                    x2 = bin_centers[i] + float(bin_step) / 2
                    cdfs = norm.cdf([x1, x2], loc=x[j], scale=sigma)
                    v[j, i] = cdfs[1] - cdfs[0]
            return v, bin_centers


def position_encoding(num_ages, embed_dim, amp_factor=1.0, n=10000):
    """
    Positional encoding for age
    """
    ages = np.arange(start=0, stop=num_ages, step=1)
    encoding_matrix = np.zeros((num_ages, embed_dim), dtype=np.float32)
    for i in range(embed_dim):
        if i % 2 == 0:
            encoding_matrix[:, i] = np.sin(ages / n ** (i / embed_dim)) * amp_factor
        else:
            encoding_matrix[:, i] = np.cos(ages / n ** ((i - 1) / embed_dim)) * amp_factor
    return encoding_matrix
