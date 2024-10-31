import lightning as lg
from models.decoder import Decoder
from models.encoder import Encoder
from models.utils import reparameterize, init_optimizer, crop_brain
from models.losses import mse, gaussian_kl, pairwise_gaussian_kl, check_weights, bce, kl_divergence
from scripts.t1_dataset import soft_label
from torch import optim, tensor, nn, isnan, cat


class ICVAE(lg.LightningModule):
    def __init__(self,
                 input_shape=(160, 192, 160),
                 latent_dim=354,
                 layers=None,
                 age_dim=0,
                 bmi_dim=46,
                 predict_age=False,
                 invariance=False,
                 lr=0.0004,
                 min_lr=0.0004,
                 optimizer='AdamW',
                 momentum=0.9,
                 weight_decay=0.01,
                 losses_weights=None):
        super(ICVAE, self).__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.predict_age = predict_age
        self.invariance = invariance
        self.lr, self.min_lr = lr, min_lr
        self.momentum, self.weight_decay = momentum, weight_decay
        check_weights(losses_weights)
        self.losses_weights = losses_weights
        self.encoder = Encoder(input_shape, latent_dim, layers)
        if invariance != 'gender':
            self.gender = nn.Sequential(nn.Linear(latent_dim, 1), nn.Sigmoid())
        if invariance != 'bmi':
            self.bmi = nn.Sequential(nn.Linear(latent_dim, bmi_dim), nn.LogSoftmax(dim=1))
        if predict_age:
            self.age = nn.Sequential(nn.Linear(latent_dim, age_dim), nn.LogSoftmax(dim=1))
        features_shape = self.encoder.final_shape
        reversed_layers = dict(reversed(layers.items()))
        conditional_dim = age_dim
        if self.invariance == 'gender':
            conditional_dim = 1
        elif self.invariance == 'bmi':
            conditional_dim = bmi_dim
        self.decoder = Decoder(features_shape, latent_dim, reversed_layers, conditional_dim, invariance)

    def forward(self, x_transformed, condition):
        mu, logvar = self.encoder(x_transformed)
        z = reparameterize(mu, logvar)
        gender = self.gender(z) if self.invariance != 'gender' else tensor(0.0)
        bmi = self.bmi(z) if self.invariance != 'bmi' else tensor(0.0)
        age = self.age(z) if self.predict_age else tensor(0.0)
        x_reconstructed = self.decoder(z, condition)
        return x_reconstructed, mu, logvar, gender, bmi, age

    def configure_optimizers(self):
        optimizer = init_optimizer(self.optimizer, self.parameters(), lr=self.lr, momentum=self.momentum,
                                    weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=(self.min_lr / self.lr),
                                                   total_iters=self.trainer.max_epochs)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'epoch'}]

    def training_step(self, batch, batch_idx):
        x, x_transformed, age, gender, bmi = batch
        condition = self.get_condition(age, gender, bmi)
        x_reconstructed, mu, logvar, gender_pred, bmi_pred, age_pred = self(x_transformed, condition)
        loss, loss_dict = self._loss(x_reconstructed, x, mu, logvar, gender_pred, bmi_pred, age_pred, gender, bmi, age)
        self.log_dict(loss_dict, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, age, gender, bmi = batch
        condition = self.get_condition(age, gender, bmi)
        x_reconstructed, mu, logvar, gender_pred, bmi_pred, age_pred = self(x, condition)
        loss, loss_dict = self._loss(x_reconstructed, x, mu, logvar, gender_pred, bmi_pred, age_pred, gender, bmi, age,
                                     mode='val')
        self.log_dict(loss_dict, sync_dist=True)
        return x_reconstructed

    def _loss(self, recon_x, x, mu, logvar, gender_pred, bmi_pred, age_pred, gender, bmi, age, mode='train'):
        recon_loss, prior_loss = mse(crop_brain(recon_x), crop_brain(x)), gaussian_kl(mu, logvar).mean()
        loss = self.losses_weights['reconstruction'] * recon_loss + self.losses_weights['prior'] * prior_loss
        marginal_loss = tensor(0.0)
        if self.invariance:
            marginal_loss = pairwise_gaussian_kl(mu, logvar, self.hparams.latent_dim).mean()
            loss += self.losses_weights['marginal'] * marginal_loss
        gender_loss = tensor(0.0)
        if self.invariance != 'gender':
            gender_loss = bce(gender_pred, gender)
            loss += self.losses_weights['gender'] * gender_loss
        bmi_loss = tensor(0.0)
        if self.invariance != 'bmi':
            nan_mask = ~isnan(bmi)
            bmi_loss = kl_divergence(bmi_pred[nan_mask], bmi[nan_mask])
            loss += self.losses_weights['bmi'] * bmi_loss
        age_loss = tensor(0.0)
        if self.predict_age:
            age_loss = kl_divergence(age_pred, age)
            loss += self.losses_weights['age'] * age_loss
        return loss, self._log_dict(mode, recon_loss.item(), prior_loss.item(), marginal_loss.item(),
                                    gender_loss.item(), bmi_loss.item(), age_loss.item())

    def _log_dict(self, mode, recon_loss, prior_loss, marginal_loss, gender_loss, bmi_loss, age_loss):
        state = {f'{mode}/recon_loss': recon_loss,
                 f'{mode}/prior_loss': prior_loss}
        if self.invariance != 'gender':
            state[f'{mode}/gender_loss'] = gender_loss
        if self.invariance != 'bmi':
            state[f'{mode}/bmi_loss'] = bmi_loss
        if self.predict_age:
            state[f'{mode}/age_loss'] = age_loss
        if self.invariance:
            state[f'{mode}/marginal_loss'] = marginal_loss
        state[f'{mode}/loss'] = recon_loss + prior_loss + marginal_loss + gender_loss + bmi_loss + age_loss
        return state

    def get_condition(self, age, gender, bmi):
        condition = None
        if self.invariance == 'age':
            condition = age
        elif self.invariance == 'gender':
            condition = gender
        elif self.invariance == 'bmi':
            nan_mask = isnan(bmi)
            if nan_mask.any():
                median = soft_label(bmi.shape[-1] / 2, 0, bmi.shape[-1]).to(bmi.device)
                repeats = bmi[nan_mask].shape[-1] // median.shape[-1]
                bmi[nan_mask] = cat([median] * repeats)
            condition = bmi
        return condition
