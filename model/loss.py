from torch import sum
import torch.nn as nn


def vae_loss(recon_x, x, mu, logvar):
    return reconstruction_loss(recon_x, x) + kl_divergence(mu, logvar)


def reconstruction_loss(recon_x, x):
    return nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')


def kl_divergence(mu, logvar):
    return -0.5 * sum(1 + logvar - mu.pow(2) - logvar.exp())
