import lightning as lg
from models.decoder import Decoder
from models.encoder import Encoder
from models.utils import reparameterize, init_optimizer
from models.losses import mse, kl_divergence, pairwise_gaussian_kl, check_weights, frange_cycle
from torch import optim, zeros
from numpy import array


class ICVAE(lg.LightningModule):
    def __init__(self,
                 input_shape=(160, 192, 160),
                 latent_dim=354,
                 layers=None,
                 conditional_dim=0,
                 lr=0.1,
                 max_lr=0.01,
                 optimizer='AdamW',
                 momentum=0.9,
                 weight_decay=0.0005,
                 beta=0.001,
                 beta_strategy='constant',
                 losses_weights=None
                 ):
        super(ICVAE, self).__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.lr, self.max_lr = lr, max_lr
        self.momentum, self.weight_decay = momentum, weight_decay
        check_weights(losses_weights)
        self.losses_weights = losses_weights
        self.encoder = Encoder(input_shape, latent_dim, layers)
        features_shape = self.encoder.final_shape
        reversed_layers = dict(reversed(layers.items()))
        self.decoder = Decoder(features_shape, latent_dim, reversed_layers, conditional_dim)
        self.beta, self.beta_strategy, self.beta_values = beta, beta_strategy, None

    def forward(self, x_transformed, condition=None):
        mu, logvar = self.encoder(x_transformed)
        z = reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z, condition)
        return x_reconstructed, mu, logvar

    def configure_optimizers(self):
        optimizer = init_optimizer(self.optimizer, self.parameters(), lr=self.lr, momentum=self.momentum,
                                    weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.max_lr,
                                                     total_steps=self.trainer.estimated_stepping_batches)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, x_transformed, condition = batch
        x_reconstructed, mu, logvar = self(x_transformed, condition)
        loss, loss_dict = self._loss(x_reconstructed, x, mu, logvar)
        self.log_dict(loss_dict, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, condition = batch
        x_reconstructed, mu, logvar = self(x, condition)
        loss, loss_dict = self._loss(x_reconstructed, x, mu, logvar, mode='val')
        self.log_dict(loss_dict, sync_dist=True)
        return x_reconstructed

    def on_train_start(self):
        if self.beta_strategy == 'cyclical':
            self.beta_values = frange_cycle(self.beta, .5, self.trainer.estimated_stepping_batches, 4, .99,
                                            mode='cosine')
        elif self.beta_strategy == 'monotonic':
            self.beta_values = frange_cycle(self.beta, .5, self.trainer.estimated_stepping_batches, 1, .99,
                                            mode='cosine')
        elif self.beta_strategy == 'constant':
            self.beta_values = array([self.beta] * self.trainer.estimated_stepping_batches)
        else:
            self.beta_values = None

    def _loss(self, recon_x, x, mu, logvar, mode='train'):
        recon_loss = mse(recon_x, x) * self.losses_weights['reconstruction']
        prior_loss = kl_divergence(mu, logvar).mean() * self.losses_weights['prior']
        if self.beta_values is not None:
            beta = self.beta_values[self.trainer.estimated_stepping_batches - 1]
            prior_loss *= beta
            self.log('beta', beta, sync_dist=True)
        loss = recon_loss + prior_loss
        marginal_loss = zeros(1)
        if self.hparams.conditional_dim > 0:
            marginal_loss = (pairwise_gaussian_kl(mu, logvar, self.hparams.latent_dim).mean()
                             * self.losses_weights['marginal'])
            loss += marginal_loss
        return loss, self._log_dict(mode, recon_loss.item(), prior_loss.item(), marginal_loss.item())

    def _log_dict(self, mode, recon_loss, prior_loss, marginal_loss):
        state = {f'{mode}_recon_loss': recon_loss,
                 f'{mode}_prior_loss': prior_loss}
        if self.hparams.conditional_dim > 0:
            state[f'{mode}_marginal_loss'] = marginal_loss
        state[f'{mode}_loss'] = recon_loss + prior_loss + marginal_loss
        return state
