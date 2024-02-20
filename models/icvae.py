from models.decoder import Decoder
from models.encoder import Encoder
from models.utils import reparameterize
from torch import optim
import lightning as lg


class ICVAE(lg.LightningModule):
    def __init__(self,
                 input_shape=(160, 192, 160),
                 latent_dim=354,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 pool_size=2,
                 pool_stride=2,
                 last_kernel_size=1,
                 last_padding=0,
                 channels=(32, 64, 128, 256, 256, 64),
                 conditional_dim=0,
                 lr=0.1,
                 max_lr=0.01,
                 optimizer='AdamW',
                 momentum=0.9,
                 weight_decay=0.0005,
                 losses_weights=None,
                 num_steps=None
                 ):
        super(ICVAE, self).__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.lr, self.max_lr = lr, max_lr
        self.momentum, self.weight_decay = momentum, weight_decay
        self.losses_weights = losses_weights
        self.num_steps = num_steps
        channels = list(channels)
        self.encoder = Encoder(input_shape, latent_dim, kernel_size, stride, padding, pool_size, pool_stride,
                                 last_kernel_size, last_padding, channels)
        features_shape = self.encoder.features_shape(input_shape)
        channels.reverse()
        self.decoder = Decoder(features_shape, latent_dim, kernel_size, stride, padding, pool_size,
                               pool_stride, last_kernel_size, last_padding, channels, conditional_dim)

    def forward(self, x, condition=None):
        mu, logvar, pooling_indices = self.encoder(x)
        z = reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z, pooling_indices, condition)
        return x_reconstructed, mu, logvar

    def configure_optimizers(self):
        if self.optimizer == 'AdamW':
            optimizer = optim.AdamW(self.parameters(), lr=self.lr, betas=(self.momentum, 0.999),
                                    weight_decay=self.weight_decay)
        elif self.optimizer == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, nesterov=True,
                                  weight_decay=self.weight_decay)
        else:
            optimizer = getattr(optim, self.optimizer)(self.parameters(), lr=self.lr)
        lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.max_lr, total_steps=self.num_steps)

        return optimizer, lr_scheduler
