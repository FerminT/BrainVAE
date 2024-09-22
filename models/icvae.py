import lightning as lg
from models.decoder import Decoder
from models.encoder import Encoder
from models.utils import reparameterize, init_optimizer, crop_brain
from models.losses import mse, kl_divergence, pairwise_gaussian_kl, check_weights
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
                 losses_weights=None):
        super(ICVAE, self).__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.invariant = invariant
        self.lr, self.min_lr = lr, min_lr
        self.momentum, self.weight_decay = momentum, weight_decay
        check_weights(losses_weights)
        self.losses_weights = losses_weights
        self.encoder = Encoder(input_shape, latent_dim, layers)
        # TODO: Add linear layers (of shape latent_dim x 2/1) for predicting gender and bmi
        features_shape = self.encoder.final_shape
        reversed_layers = dict(reversed(layers.items()))
        self.decoder = Decoder(features_shape, latent_dim, reversed_layers, conditional_dim)

    def forward(self, x_transformed, condition):
        mu, logvar = self.encoder(x_transformed)
        z = reparameterize(mu, logvar)
        # TODO: forward the embedding to linear layers for predicting gender and bmi
        x_reconstructed = self.decoder(z, condition)
        return x_reconstructed, mu, logvar

    def configure_optimizers(self):
        optimizer = init_optimizer(self.optimizer, self.parameters(), lr=self.lr, momentum=self.momentum,
                                    weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=(self.min_lr / self.lr),
                                                   total_iters=self.trainer.max_epochs)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'epoch'}]

    def training_step(self, batch, batch_idx):
        x, x_transformed, condition, gender, bmi = batch
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

    def _loss(self, recon_x, x, mu, logvar, mode='train'):
        recon_loss, prior_loss = mse(crop_brain(recon_x), crop_brain(x)), kl_divergence(mu, logvar).mean()
        recon_loss_value, prior_loss_value = recon_loss.item(), prior_loss.item()
        recon_loss *= self.losses_weights['reconstruction']
        prior_loss *= self.losses_weights['prior']
        loss = recon_loss + prior_loss
        marginal_loss_value = 0.0
        if self.invariant:
            marginal_loss = pairwise_gaussian_kl(mu, logvar, self.hparams.latent_dim).mean()
            marginal_loss_value = marginal_loss.item()
            marginal_loss *= self.losses_weights['marginal']
            loss += marginal_loss
        return loss, self._log_dict(mode, recon_loss_value, prior_loss_value, marginal_loss_value)

    def _log_dict(self, mode, recon_loss, prior_loss, marginal_loss):
        state = {f'{mode}_recon_loss': recon_loss,
                 f'{mode}_prior_loss': prior_loss}
        if self.invariant:
            state[f'{mode}_marginal_loss'] = marginal_loss
        state[f'{mode}_loss'] = recon_loss + prior_loss + marginal_loss
        return state
