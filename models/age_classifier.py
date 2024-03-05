""" Fully connected neural network that takes the latent representation of the MRI data as input
    and predicts the age of the subject as output."""

import lightning as lg
from models.icvae import ICVAE
from models.utils import reparameterize, init_optimizer
from torch import nn, optim


class AgeClassifier(lg.LightningModule):

    def __init__(self,
                 encoder_path,
                 input_dim=354,
                 output_dim=1,
                 hidden_dims=(128, 64, 32),
                 lr=0.1,
                 optimizer='SGD',
                 momentum=0.9,
                 weight_decay=0.0005,
                 step_size=5):
        super(AgeClassifier, self).__init__()
        self.save_hyperparameters()
        self.encoder = ICVAE.load_from_checkpoint(encoder_path).encoder
        self.encoder.eval()
        self.fc_layers = create_fc_layers(input_dim, output_dim, hidden_dims)
        self.lr, self.optimizer = lr, optimizer
        self.momentum, self.weight_decay, self.step_size = momentum, weight_decay, step_size

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        return self.fc_layers(z)

    def configure_optimizers(self):
        optimizer = init_optimizer(self.optimizer, self.parameters(), lr=self.lr, momentum=self.momentum,
                                   weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=0.1)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, age = batch
        prediction = self(x)
        loss = nn.functional.l1_loss(prediction, age)
        self.log('train_mae', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, age = batch
        prediction = self(x)
        loss = nn.functional.l1_loss(prediction, age)
        self.log('val_mae', loss)
        return loss


def create_fc_layers(input_dim, output_dim, hidden_dims):
    layers = list()
    dims = [input_dim] + list(hidden_dims) + [output_dim]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)
