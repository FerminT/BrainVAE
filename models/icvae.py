import lightning as lg
from models.decoder import Decoder
from models.encoder import Encoder
from models.utils import reparameterize, init_optimizer
from scripts.utils import crop_brain
from models.losses import mse, kl_divergence, pairwise_gaussian_kl, check_weights, frange_cycle, step_cycle
from torch import optim


class ICVAE(lg.LightningModule):
    def __init__(self,
                 input_shape=(160, 192, 160),
                 latent_dim=354,
                 layers=None,
                 conditional_dim=0,
                 invariant=False,
                 lr=0.0004,
                 min_lr=0.0004,
                 optimizer='AdamW',
                 momentum=0.9,
                 weight_decay=0.01,
                 beta=0.0001,
                 beta_strategy='stepwise',
                 losses_weights=None
                 ):
        super(ICVAE, self).__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.invariant = invariant
        self.lr, self.min_lr = lr, min_lr
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
        lr_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=(self.min_lr / self.lr),
                                                   total_iters=self.trainer.max_epochs)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'epoch'}]

    def training_step(self, batch, batch_idx):
        x, x_transformed, condition = batch
        condition = condition if self.invariant else None
        x_reconstructed, mu, logvar = self(x_transformed, condition)
        loss, loss_dict = self._loss(x_reconstructed, x, mu, logvar)
        self.log_dict(loss_dict, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, condition = batch
        condition = condition if self.invariant else None
        x_reconstructed, mu, logvar = self(x, condition)
        loss, loss_dict = self._loss(x_reconstructed, x, mu, logvar, mode='val')
        self.log_dict(loss_dict, sync_dist=True)
        return x_reconstructed

    def on_train_start(self):
        if self.beta_strategy == 'cyclical':
            self.beta_values = frange_cycle(self.beta, 1.0, self.trainer.estimated_stepping_batches, 4, .99,
                                            mode='cosine')
        elif self.beta_strategy == 'monotonic':
            self.beta_values = frange_cycle(self.beta, 1.0, self.trainer.estimated_stepping_batches, 1, .99,
                                            mode='cosine')
        elif self.beta_strategy == 'stepwise':
            self.beta_values = step_cycle(self.beta, self.trainer.estimated_stepping_batches)
        else:
            self.beta_values = None

    def _loss(self, recon_x, x, mu, logvar, mode='train'):
        recon_loss, prior_loss = mse(crop_brain(recon_x), crop_brain(x)), kl_divergence(mu, logvar).mean()
        recon_loss_value, prior_loss_value = recon_loss.item(), prior_loss.item()
        recon_loss *= self.losses_weights['reconstruction']
        prior_loss *= self.losses_weights['prior']
        if self.beta_values is not None:
            beta = self.beta_values[min(len(self.beta_values) - 1, self.trainer.global_step)]
            prior_loss *= beta
            self.log('beta', beta * self.losses_weights['prior'], sync_dist=True)
        loss = recon_loss + prior_loss
        marginal_loss_value = 0.0
        if self.invariant:
            marginal_loss = pairwise_gaussian_kl(mu, logvar, self.hparams.latent_dim).mean()
            marginal_loss_value = marginal_loss.item()
            marginal_loss *= self.losses_weights['marginal']
            if self.beta_values is not None:
                beta = self.beta_values[min(len(self.beta_values) - 1, self.trainer.global_step)]
                marginal_loss *= beta
            loss += marginal_loss
        return loss, self._log_dict(mode, recon_loss_value, prior_loss_value, marginal_loss_value)

    def _log_dict(self, mode, recon_loss, prior_loss, marginal_loss):
        state = {f'{mode}_recon_loss': recon_loss,
                 f'{mode}_prior_loss': prior_loss}
        if self.hparams.conditional_dim > 0:
            state[f'{mode}_marginal_loss'] = marginal_loss
        state[f'{mode}_loss'] = recon_loss + prior_loss + marginal_loss
        return state
