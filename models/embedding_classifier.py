import lightning as lg
from models.utils import init_optimizer
from torch import nn, optim, exp, cat


class EmbeddingClassifier(lg.LightningModule):

    def __init__(self,
                 input_dim=354,
                 output_dim=1,
                 hidden_dims=(128, 64, 32),
                 n_layers=3,
                 lr=0.001,
                 optimizer='AdamW',
                 momentum=0.9,
                 weight_decay=0.0005,
                 bin_centers=None,
                 use_age=False):
        super(EmbeddingClassifier, self).__init__()
        self.save_hyperparameters()
        self.fc_layers = create_fc_layers(input_dim, output_dim, hidden_dims, n_layers)
        self.lr, self.optimizer, self.output_dim = lr, optimizer, output_dim
        self.momentum, self.weight_decay = momentum, weight_decay
        self.bin_centers = bin_centers
        self.use_age = use_age

    def configure_optimizers(self):
        optimizer = init_optimizer(self.optimizer, self.parameters(), lr=self.lr, momentum=self.momentum,
                                   weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1,
                                                   total_iters=self.trainer.estimated_stepping_batches // 10)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def forward(self, z, age=None):
        if self.use_age:
            z = cat([z, age], dim=1)
        return self.fc_layers(z)

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    def _step(self, batch, mode):
        z, targets, ages = batch
        predictions = self(z, ages)
        if self.output_dim == 1:
            loss = nn.functional.binary_cross_entropy_with_logits(predictions, targets)
            self.log(f'{mode}_bce', loss.item())
            self.log(f'{mode}_accuracy', ((predictions > 0.5) == targets).float().mean().item())
        else:
            loss = nn.functional.kl_div(predictions, targets, reduction='batchmean')
            predicted_values = exp(predictions.float().cpu().detach()) @ self.bin_centers
            target_values = targets.float().cpu() @ self.bin_centers
            self.log(f'{mode}_mae', abs(predicted_values - target_values).mean())
            self.log(f'{mode}_prediction', predicted_values.mean().item())
        return loss


def create_fc_layers(input_dim, output_dim, hidden_dims, n_layers):
    layers = list()
    if n_layers > 3 or n_layers < 0:
        raise ValueError('Number of layers must be between 0 and 3')
    hidden_dims = hidden_dims[(-1) * n_layers:] if n_layers > 0 else []
    dims = [input_dim] + list(hidden_dims) + [output_dim]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
        if i == len(dims) - 2 and output_dim > 1:
            layers.append(nn.LogSoftmax(dim=1))
    return nn.Sequential(*layers)
