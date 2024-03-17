from torch import matmul, unsqueeze
from numpy import ones
from math import pi, cos
import torch.nn as nn


def check_weights(weights):
    if weights is None or 'reconstruction' not in weights or 'prior' not in weights or 'marginal' not in weights:
        raise ValueError('Loss weights dict must contain keys: reconstruction, prior and marginal')


def mse(recon_x, x):
    return nn.functional.mse_loss(recon_x, x, reduction='mean')


def l1(recon_x, x):
    return nn.functional.l1_loss(recon_x, x, reduction='mean')


def kl_divergence(mu, logvar):
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(axis=1)


def pairwise_gaussian_kl(mu, logvar, latent_dim):
    """ Pairwise Gaussian KL divergence, from Moyer et al. 2018
        Used as an approximation of KL[(q(z|x) || q(z))] """
    sigma_sq = logvar.exp()
    sigma_sq_inv = 1.0 / sigma_sq
    first_term = matmul(sigma_sq, sigma_sq_inv.transpose(0, 1))

    r = matmul(mu * mu, sigma_sq_inv.transpose(0, 1))
    r2 = (mu * mu * sigma_sq_inv).sum(axis=1)
    second_term = 2 * matmul(mu, (mu * sigma_sq_inv).transpose(0, 1))
    second_term = r - second_term + r2
    det_sigma = logvar.sum(axis=1)
    third_term = unsqueeze(det_sigma, dim=1) - unsqueeze(det_sigma, dim=1).transpose(0, 1)

    return 0.5 * (first_term + second_term + third_term - latent_dim)


def frange_cycle(start, stop, total_steps, n_cycle, ratio, mode='linear'):
    beta_at_steps = ones(total_steps)
    period = total_steps / n_cycle
    step = (stop - start) / (period * ratio)
    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop:
            if mode == 'linear':
                beta_at_steps[int(i + c * period)] = v
            else:
                beta_at_steps[int(i + c * period)] = 0.5 - .5 * cos(v * pi)
            v += step
            i += 1
