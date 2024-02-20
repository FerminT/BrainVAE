import yaml
import numpy as np
from torch import cat, optim, nn as nn
from torchvision.utils import save_image
from scipy.stats import norm
from models import icvae


def load_architecture(model_name, config, num_batches, num_epochs, device):
    config['num_steps'] = num_batches * num_epochs
    model = getattr(icvae, model_name.upper())(**config)
    model.to(device)
    return model


def save_reconstruction_batch(data, recon_batch, epoch, save_path):
    n, n_slice = min(data.size(0), 8), 50
    imgs_path = save_path / 'reconstructions'
    imgs_path.mkdir(exist_ok=True)
    for axis in range(3):
        if axis == 0:
            original_slice = data[:, :, n_slice, :, :]
            reconstructed_slice = recon_batch[:, :, n_slice, :, :]
        elif axis == 1:
            original_slice = data[:, :, :, n_slice, :]
            reconstructed_slice = recon_batch[:, :, :, n_slice, :]
        else:
            original_slice = data[:, :, :, :, n_slice]
            reconstructed_slice = recon_batch[:, :, :, :, n_slice]
        comparison = cat([original_slice[:n], reconstructed_slice[:n]])
        save_image(comparison.cpu(), imgs_path / f'epoch_{epoch}_axis_{axis}.png', nrow=n)


def load_yaml(filepath):
    with filepath.open('r') as file:
        return yaml.safe_load(file)


def num2vect(x, bin_range, bin_step, sigma):
    """
    bin_range: (start, end), size-2 tuple
    bin_step: should be a divisor of |end-start|
    sigma:
    = 0 for 'hard label', in which case v is an index
    > 0 for 'soft label', in which case v is a vector
    """
    bin_start = bin_range[0]
    bin_end = bin_range[1]
    bin_length = bin_end - bin_start
    if not bin_length % bin_step == 0:
        raise ValueError("bin's range should be divisible by bin_step!")
    bin_number = int(bin_length / bin_step)
    bin_centers = bin_start + float(bin_step) / 2 + bin_step * np.arange(bin_number)

    if sigma == 0:
        x = np.array(x, dtype=np.float32)
        i = np.floor((x - bin_start) / bin_step)
        i = i.astype(int)
        return i, bin_centers
    elif sigma > 0:
        if np.isscalar(x):
            v = np.zeros((bin_number,), dtype=np.float32)
            for i in range(bin_number):
                x1 = bin_centers[i] - float(bin_step) / 2
                x2 = bin_centers[i] + float(bin_step) / 2
                cdfs = norm.cdf([x1, x2], loc=x, scale=sigma)
                v[i] = cdfs[1] - cdfs[0]
            return v, bin_centers
        else:
            v = np.zeros((len(x), bin_number), dtype=np.float32)
            for j in range(len(x)):
                for i in range(bin_number):
                    x1 = bin_centers[i] - float(bin_step) / 2
                    x2 = bin_centers[i] + float(bin_step) / 2
                    cdfs = norm.cdf([x1, x2], loc=x[j], scale=sigma)
                    v[j, i] = cdfs[1] - cdfs[0]
            return v, bin_centers
