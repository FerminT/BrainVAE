from torch import matmul, unsqueeze, zeros
import torch.nn as nn


class Loss:

    def __init__(self, dataset_size, latent_dim, conditional_dim, best_loss=float('inf')):
        self.mode = 'train'
        self.dataset_size = dataset_size
        self.latent_dim = latent_dim
        self.is_conditional = conditional_dim > 0
        self.best_loss = best_loss
        self.is_best = False
        self.avg_recon_loss, self.avg_prior_loss, self.avg_marginal_loss = 0, 0, 0

    def __call__(self, recon_x, x, mu, logvar):
        recon_loss = mse(recon_x, x)
        prior_loss = kl_divergence(mu, logvar).mean()
        loss = recon_loss + prior_loss
        self.avg_recon_loss += recon_loss.item() / self.dataset_size
        self.avg_prior_loss += prior_loss.item() / self.dataset_size
        marginal_loss = zeros(1)
        if self.is_conditional:
            marginal_loss = pairwise_gaussian_kl(mu, logvar, self.latent_dim).mean()
            loss += marginal_loss
            self.avg_marginal_loss += marginal_loss.item() / self.dataset_size

        return loss, log_dict(self.mode, recon_loss.item(), prior_loss.item(), marginal_loss.item(), step='batch')

    def get_avg(self):
        return self.avg_recon_loss + self.avg_prior_loss + self.avg_marginal_loss

    def step(self):
        if self.mode == 'val':
            if self.get_avg() < self.best_loss:
                self.best_loss = self.get_avg()
                self.is_best = True
            else:
                self.is_best = False

    def state_dict(self):
        return log_dict(self.mode, self.avg_recon_loss, self.avg_prior_loss, self.avg_marginal_loss, step='epoch')

    def reset(self):
        self.avg_recon_loss, self.avg_prior_loss, self.avg_marginal_loss = 0, 0, 0

    def eval(self):
        self.reset()
        self.mode = 'val'

    def train(self):
        self.reset()
        self.mode = 'train'


def log_dict(mode, recon_loss, prior_loss, marginal_loss, step):
    state = {f'{mode}/{step}_recon_loss': recon_loss,
             f'{mode}/{step}_prior_loss': prior_loss}
    if marginal_loss != 0:
        state[f'{mode}/{step}_marginal_loss'] = marginal_loss
    return state


def mse(recon_x, x):
    return nn.functional.mse_loss(recon_x, x, reduction='mean')


def kl_divergence(mu, logvar):
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(axis=1)


def pairwise_gaussian_kl(mu, logvar, latent_dim):
    """ Pairwise Gaussian KL divergence, from Moyer et al. 2018
        Used as an approximation of KL[(q(z|x) || q(z))] """
    sigma_sq = logvar.exp()
    sigma_sq_inv = 1.0 / sigma_sq
    first_term = matmul(sigma_sq, sigma_sq_inv.transpose(0, 1))

    r = matmul(mu * mu, sigma_sq_inv.squeeze().transpose(0, 1))
    r2 = (mu * mu * sigma_sq_inv.squeeze()).sum(axis=1)
    second_term = 2 * matmul(mu, (mu * sigma_sq_inv.squeeze()).transpose(0, 1))
    second_term = r - second_term + r2
    det_sigma = logvar.sum(axis=1)
    third_term = unsqueeze(det_sigma, dim=1) - unsqueeze(det_sigma, dim=1).transpose(0, 1)

    return 0.5 * (first_term + second_term + third_term - latent_dim)
