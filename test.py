from model import icvae
from torch import randn


def encoder_decoder_shapes():
    input_shape, latent_dim = (160, 192, 160), 354
    encoder = icvae.Encoder(input_shape=input_shape)
    x = randn((1, 1, *input_shape))
    mu, logvar, pooling_indices = encoder(x)
    assert mu.shape == (1, latent_dim), f"mu.shape: {mu.shape}"
    assert logvar.shape == (1, latent_dim), f"logvar.shape: {logvar.shape}"
    assert len(pooling_indices) == encoder.n_blocks - 1, f"len(pooling_indices): {len(pooling_indices)}"

    decoder = icvae.Decoder(latent_dim=latent_dim)
    z = icvae.reparameterize(mu, logvar)
    x_recon = decoder(z, pooling_indices)
    assert x_recon.shape == (1, 1, *input_shape), f"x_recon.shape: {x_recon.shape}"
    print("encoder_decoder_shapes passed")


def forward_pass():
    input_shape, latent_dim = (160, 192, 160), 354
    vae = icvae.ICVAE(input_shape=input_shape, latent_dim=latent_dim)
    x = randn((1, 1, *input_shape))
    x_recon, mu, logvar = vae(x)
    assert x_recon.shape == (1, 1, *input_shape), f"x_recon.shape: {x_recon.shape}"
    assert mu.shape == (1, latent_dim), f"mu.shape: {mu.shape}"
    assert logvar.shape == (1, latent_dim), f"logvar.shape: {logvar.shape}"
    print("forward_pass passed")


if __name__ == "__main__":
    encoder_decoder_shapes()
    forward_pass()
    print("All tests passed")
