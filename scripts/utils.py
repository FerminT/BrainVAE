import pandas as pd
import yaml
from sklearn.manifold import MDS, TSNE, Isomap
from sklearn.decomposition import PCA
from torch import cat, device, cuda
from torchvision.utils import make_grid
from torchvision.transforms import Resize
from tqdm import tqdm
from models.utils import get_latent_representation
from models.icvae import ICVAE
from scripts.t1_dataset import T1Dataset
from scripts.data_handler import load_set
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


def load_yaml(filepath):
    with filepath.open('r') as file:
        return yaml.safe_load(file)


def load_model(weights_path, device):
    weights = get_weights(weights_path)
    model = ICVAE.load_from_checkpoint(weights).to(device)
    model.eval()
    return model


def get_weights(weights_path):
    return next(weights_path.parent.glob(f'{weights_path.name}*'))


def subjects_embeddings(weights_path, model_name, dataset_name, input_shape, latent_dim, split, datapath, splits_path,
                        random_state, save_path):
    data, age_range, bmi_range = load_set(dataset_name, split, splits_path, random_state)
    dataset = T1Dataset(input_shape, datapath, data, latent_dim, age_dim=1, age_range=age_range, bmi_range=bmi_range,
                        testing=True)
    device_ = device('cuda' if cuda.is_available() else 'cpu')
    filename = save_path / f'subjects_embeddings.pkl'
    if filename.exists():
        return pd.read_pickle(filename)
    model = load_model(weights_path, device_) if model_name != 'age' else 'age'
    subjects = []
    print('Computing embeddings...')
    for idx in tqdm(range(len(dataset))):
        t1_img, _, age, _, _ = dataset[idx]
        t1_img = t1_img.unsqueeze(dim=0).to(device_)
        z = get_latent_representation(t1_img, model.encoder) if model_name != 'age' else age.unsqueeze(dim=0)
        subject_metadata = dataset.get_metadata(idx).copy()
        subject_metadata['embedding'] = z.cpu().detach().squeeze(dim=0).numpy()
        subjects.append(subject_metadata)
    subjects_df = pd.DataFrame(subjects).set_index('subject_id')
    subjects_df.to_pickle(filename)
    return subjects_df


def init_embedding(method, n_components=2):
    if method == 'mds':
        embedding = MDS(n_components=n_components,
                        random_state=42)
    elif method == 'tsne':
        embedding = TSNE(n_components=n_components,
                         perplexity=30,
                         init='pca',
                         random_state=42)
    elif method == 'isomap':
        embedding = Isomap(n_components=n_components,
                           n_neighbors=10,
                           n_jobs=-1)
    elif method == 'pca':
        embedding = PCA(n_components=n_components)
    elif method == 'umap':
        embedding = umap.UMAP(n_components=n_components,
                              random_state=42)
    else:
        raise NotImplementedError(f'Method {method} not implemented')

    return embedding
