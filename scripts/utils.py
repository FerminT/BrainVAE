import yaml
import numpy as np
from torch import cat
from torchvision.utils import make_grid
from torchvision.transforms import Resize
from scipy.stats import norm
from models.icvae import ICVAE
from scripts.constants import SPLITS_PATH


def load_architecture(config, num_batches, num_epochs):
    config['num_steps'] = num_batches * num_epochs
    model = ICVAE(**config)
    return model


def reconstruction_comparison_grid(data, outputs, n, slice_idx, epoch):
    imgs, captions = [], []
    max_shape = [0, 0]
    for axis in range(3):
        if axis == 0:
            original_slice = data[:, :, slice_idx, :, :]
            reconstructed_slice = outputs[:, :, slice_idx, :, :]
        elif axis == 1:
            original_slice = data[:, :, :, slice_idx, :]
            reconstructed_slice = outputs[:, :, :, slice_idx, :]
        else:
            original_slice = data[:, :, :, :, slice_idx]
            reconstructed_slice = outputs[:, :, :, :, slice_idx]
        img_comparison = make_grid(cat([original_slice[:n], reconstructed_slice[:n]]), nrow=n)
        imgs.append(img_comparison)
        captions.append(f'Epoch: {epoch} Axis: {axis}')
        max_shape[0] = max(max_shape[0], img_comparison.shape[1])
        max_shape[1] = max(max_shape[1], img_comparison.shape[2])
    for i in range(len(imgs)):
        imgs[i] = Resize(max_shape, antialias=True)(imgs[i])
    return imgs, captions


def get_splits_files(datapath, sample_size):
    splits_path = datapath / SPLITS_PATH
    if sample_size != -1:
        splits_path = splits_path / f'sample_{sample_size}'
    train_csv, val_csv, test_csv = splits_path / 'train.csv', splits_path / 'val.csv', splits_path / 'test.csv'
    return train_csv, val_csv, test_csv


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
