""" Fully connected neural network that takes the latent representation of the MRI data as input
    and predicts the age of the subject as output."""

import lightning as lg
from models.utils import init_optimizer
from torch import nn, optim


class EmbeddingClassifier(lg.LightningModule):

    def __init__(self,
                 input_dim=354,
                 output_dim=1,
                 hidden_dims=(128, 64, 32),
                 lr=0.1,
                 optimizer='AdamW',
                 momentum=0.9,
                 weight_decay=0.0005,
                 data_type='continuous'):
        super(EmbeddingClassifier, self).__init__()
        self.save_hyperparameters()
        self.fc_layers = create_fc_layers(input_dim, output_dim, hidden_dims, data_type)
        self.lr, self.optimizer = lr, optimizer
        self.data_type = data_type
        self.momentum, self.weight_decay = momentum, weight_decay

    def forward(self, z):
        return self.fc_layers(z)

    def configure_optimizers(self):
        optimizer = init_optimizer(self.optimizer, self.parameters(), lr=self.lr, momentum=self.momentum,
                                   weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01,
                                                     total_steps=self.trainer.estimated_stepping_batches)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    def _step(self, batch, mode):
        z, target = batch
        prediction = self(z)
        if self.data_type == 'categorical':
            loss = nn.functional.binary_cross_entropy(prediction, target)
            self.log(f'{mode}_bce', loss.item(), sync_dist=True)
            self.log(f'{mode}_accuracy', ((prediction > 0.5) == target).float().mean().item(), sync_dist=True)
        else:
            loss = nn.functional.l1_loss(prediction, target)
            self.log(f'{mode}_mae', loss.item(), sync_dist=True)
            self.log(f'{mode}_prediction', prediction.mean().item(), sync_dist=True)
        return loss


def create_fc_layers(input_dim, output_dim, hidden_dims, data_type):
    layers = list()
    dims = [input_dim] + list(hidden_dims) + [output_dim]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
        if data_type == 'categorical' and i == len(dims) - 2:
            layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)
