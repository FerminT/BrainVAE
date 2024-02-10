import models.decoder
import models.encoder
import models.utils
from models import vae, losses
from torch import randn
from scripts.utils import load_yaml
from pathlib import Path
from torch import optim, device
from scripts.data_handler import load_datasets, get_loader

CFG_PATH = Path('cfg')
DATA_PATH = Path('datasets')


def load_ukbb_data():
    config = load_yaml(CFG_PATH / 'default.yaml')
    train, val, test = load_datasets(DATA_PATH / 'ukbb', config['params']['input_shape'], -1, 0.2, 0.1,
                                        redo_splits=False, device=device('cpu'), shuffle=True, random_state=42)
    train_loader = get_loader(train, 8, shuffle=False)
    val_loader = get_loader(val, 8, shuffle=False)
    test_loader = get_loader(test, 8, shuffle=False)
    assert len(train) > 0, f"len(train): {len(train)}"
    assert len(val) > 0, f"len(val): {len(val)}"
    assert len(test) > 0, f"len(test): {len(test)}"
    assert len(train_loader) > 0, f"len(train_loader): {len(train_loader)}"
    assert len(val_loader) > 0, f"len(val_loader): {len(val_loader)}"
    assert len(test_loader) > 0, f"len(test_loader): {len(test_loader)}"
    (DATA_PATH / 'ukbb' / 'splits' / 'train.csv').unlink()
    (DATA_PATH / 'ukbb' / 'splits' / 'val.csv').unlink()
    (DATA_PATH / 'ukbb' / 'splits' / 'test.csv').unlink()
    (DATA_PATH / 'ukbb' / 'splits').rmdir()
    print("load_ukbb_data passed")


def load_model_with_config():
    config = load_yaml(CFG_PATH / 'default.yaml')
    model = getattr(vae, 'VAE')(**config['params'])
    optimizer = getattr(optim, config['optimizer'])(model.parameters(), lr=0.001)
    criterion = getattr(losses, config['loss'])
    print("load_model_with_config passed")


def encoder_decoder_shapes():
    input_shape, latent_dim = (160, 192, 160), 354
    encoder = models.encoder.Encoder(input_shape=input_shape)
    x = randn((1, 1, *input_shape))
    mu, logvar, pooling_indices = encoder(x)
    assert mu.shape == (1, latent_dim), f"mu.shape: {mu.shape}"
    assert logvar.shape == (1, latent_dim), f"logvar.shape: {logvar.shape}"
    assert len(pooling_indices) == encoder.n_blocks - 1, f"len(pooling_indices): {len(pooling_indices)}"

    decoder = models.decoder.Decoder(latent_dim=latent_dim)
    z = models.utils.reparameterize(mu, logvar)
    x_recon = decoder(z, pooling_indices)
    assert x_recon.shape == (1, 1, *input_shape), f"x_recon.shape: {x_recon.shape}"
    print("encoder_decoder_shapes passed")


def forward_pass():
    input_shape, latent_dim = (160, 192, 160), 354
    model = vae.VAE(input_shape=input_shape, latent_dim=latent_dim)
    x = randn((1, 1, *input_shape))
    x_recon, mu, logvar = model(x)
    assert x_recon.shape == (1, 1, *input_shape), f"x_recon.shape: {x_recon.shape}"
    assert mu.shape == (1, latent_dim), f"mu.shape: {mu.shape}"
    assert logvar.shape == (1, latent_dim), f"logvar.shape: {logvar.shape}"
    print("forward_pass passed")


if __name__ == "__main__":
    encoder_decoder_shapes()
    forward_pass()
    load_model_with_config()
    load_ukbb_data()
    print("All tests passed")
