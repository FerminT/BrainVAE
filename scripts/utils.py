import pandas as pd
import yaml
import numpy as np
from sklearn.manifold import MDS, TSNE, Isomap
from sklearn.decomposition import PCA
from pandas import read_csv
from torch import cat
from torchvision.utils import make_grid
from torchvision.transforms import Resize
from scipy.stats import norm
from tqdm import tqdm
from models.utils import get_latent_representation
from scripts.constants import SPLITS_PATH
import umap


def slice_data(data, slice_idx, axis):
    if axis == 0:
        return data[:, :, slice_idx, :, :]
    elif axis == 1:
        return data[:, :, :, slice_idx, :]
    else:
        return data[:, :, :, :, slice_idx]


def reconstruction_comparison_grid(data, outputs, n, slice_idx, epoch):
    imgs, captions = [], []
    max_shape = [0, 0]
    for axis in range(3):
        original_slice = slice_data(data, slice_idx, axis)
        reconstructed_slice = slice_data(outputs, slice_idx, axis)
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


def num2vect(x, bin_range, bin_step=1, sigma=1):
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


def load_set(datapath, sample_size, split):
    train_csv, val_csv, test_csv = get_splits_files(datapath, sample_size)
    if not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
        raise ValueError(f'splits files for a sample size of {sample_size} do not exist')
    if split == 'val':
        data = read_csv(val_csv)
    elif split == 'test':
        data = read_csv(test_csv)
    else:
        data = read_csv(train_csv)
    return data


def subjects_embeddings(dataset, model, device, save_path):
    filename = save_path / 'subjects_embeddings.pkl'
    if filename.exists():
        return pd.read_pickle(filename)
    subjects = []
    print('Computing embeddings...')
    for idx in tqdm(range(len(dataset))):
        t1_img, _, _ = dataset[idx]
        t1_img = t1_img.unsqueeze(dim=0).to(device)
        z = get_latent_representation(t1_img, model.encoder)
        subject_metadata = dataset.get_metadata(idx).copy()
        subject_metadata['embedding'] = z.cpu().detach().squeeze().numpy()
        subjects.append(subject_metadata)
    subjects_df = pd.DataFrame(subjects).set_index('subject_id')
    subjects_df.to_pickle(filename)
    return subjects_df


def init_embedding(method):
    if method == 'mds':
        embedding = MDS(n_components=2,
                        random_state=42)
    elif method == 'tsne':
        embedding = TSNE(n_components=2,
                         perplexity=30,
                         init='pca',
                         random_state=42)
    elif method == 'isomap':
        embedding = Isomap(n_components=2,
                           n_neighbors=10,
                           n_jobs=-1)
    elif method == 'pca':
        embedding = PCA(n_components=2)
    elif method == 'umap':
        embedding = umap.UMAP(n_components=2,
                              random_state=42)
    else:
        raise NotImplementedError(f'Method {method} not implemented')

    return embedding
