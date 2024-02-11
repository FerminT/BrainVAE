from models.decoder import Decoder
from models.encoder import Encoder
from models.utils import reparameterize
import torch.nn as nn


class ICVAE(nn.Module):
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
                 conditional_dim=0):
        super(ICVAE, self).__init__()
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
