from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from pandas import read_csv
from scipy.stats import sem, pearsonr
from sklearn.manifold import MDS, TSNE, Isomap
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, accuracy_score
from torch import cat, device, cuda
from torchvision.utils import make_grid
from torchvision.transforms import Resize
from tqdm import tqdm
from models.utils import get_latent_representation
from models.icvae import ICVAE
from scripts.constants import CFGS_RENAMING
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


def load_model(weights_path, config, device):
    weights = get_weights(weights_path)
    model = ICVAE.load_from_checkpoint(weights, **config).to(device)
    model.eval()
    return model


def get_weights(weights_path):
    return next(weights_path.parent.glob(f'{weights_path.name}*'))


def subjects_embeddings(weights_path, model_name, dataset_name, config, split, datapath, splits_path,
                        random_state, save_path):
    data, age_range, bmi_range = load_set(dataset_name, split, splits_path, random_state)
    dataset = T1Dataset(config['input_shape'], datapath, data, config['latent_dim'], age_dim=1, age_range=age_range,
                        bmi_range=bmi_range, testing=True)
    device_ = device('cuda' if cuda.is_available() else 'cpu')
    filename = save_path / f'subjects_embeddings.pkl'
    if filename.exists():
        return pd.read_pickle(filename)
    model = load_model(weights_path, config, device_) if model_name != 'age' else None
    subjects = []
    print('Computing embeddings...')
    for idx in tqdm(range(len(dataset))):
        t1_img, _, age, _, _ = dataset[idx]
        t1_img = t1_img.unsqueeze(dim=0).to(device_)
        normalized_age = (age - age_range[0]) / (age_range[1] - age_range[0])
        z = get_latent_representation(t1_img, model.encoder) if model_name != 'age' else normalized_age.unsqueeze(dim=0)
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


def load_predictions(target_labels, cfgs, results_path):
    evaluated_cfgs = []
    labels_results = {label: {} for label in target_labels}
    for cfg in cfgs:
        cfg_path = Path(results_path, cfg)
        if len(Path(cfg).parts) == 1:
            models = [dir_ for dir_ in cfg_path.iterdir() if dir_.is_dir()]
            if len(models) == 0:
                raise ValueError(f'No models found in {cfg_path}')
            model = models[-1]
            name = CFGS_RENAMING.get(cfg, cfg)
            evaluated_cfgs.append(name)
        else:
            model = cfg_path
            name = CFGS_RENAMING.get(model.parent.name, model.parent.name)
            evaluated_cfgs.append(name)
        for label in target_labels:
            results = read_csv(Path(model, f'{label}_predictions.csv'))
            labels_results[label][name] = results
    return labels_results, evaluated_cfgs


def get_age_windows(labels_predictions, target_labels, evaluated_cfgs, age_windows):
    age_windows_ranges = {label: {} for label in target_labels}
    if age_windows > 0:
        for label in target_labels:
            for model in evaluated_cfgs:
                model_at_label = labels_predictions[label][model]
                model_at_label['age_window'] = pd.qcut(model_at_label['age_at_scan'], age_windows, labels=False)
                age_windows_ranges[label] = {f'window_{i}': (model_at_label[model_at_label['age_window'] == i]
                                                             ['age_at_scan'].min(),
                                                             model_at_label[model_at_label['age_window'] == i]
                                                             ['age_at_scan'].max())
                                              for i in range(age_windows)}
    return age_windows_ranges


def compute_metrics(labels_results, target_labels, evaluated_cfgs):
    metrics = {label: {} for label in target_labels}
    for label in target_labels:
        for model in evaluated_cfgs:
            model_results = labels_results[label][model]
            mse_list, mae_list, corr_list, pvalues_list = [], [], [], []
            acc_list = []
            for run in model_results.columns:
                if run.startswith('pred_'):
                    predictions = model_results[run].values
                    true_values = model_results['label'].values

                    if not np.array_equal(true_values, true_values.astype(bool)):
                        mae = mean_absolute_error(true_values, predictions)
                        corr = pearsonr(true_values, predictions)
                        mae_list.append(mae), corr_list.append(corr[0]), pvalues_list.append(corr[1])
                    else:
                        acc = accuracy_score(true_values, predictions >= 0.5)
                        acc_list.append(acc)
                if label == 'reconstruction_error' and run.startswith('reconstruction_error'):
                    mse_list = model_results[run].values

            if mae_list:
                metrics[label][model] = {'MAE_mean': np.mean(mae_list), 'MAE_stderr': sem(mae_list),
                                         'Correlation_mean': np.mean(corr_list) if np.mean(corr_list) > 0.01 else 0.01,
                                         'Correlation_stderr': sem(corr_list)}
                print(f'{model} {label} MAE: {np.mean(mae_list):.4f} Correlation: {np.mean(corr_list):.4f} (p: '
                      f'{np.mean(pvalues_list):.4f})')
            if acc_list:
                metrics[label][model] = {'Accuracy_mean': np.mean(acc_list), 'Accuracy_stderr': sem(acc_list)}
                print(f'{model} {label} Accuracy: {np.mean(acc_list):.4f}')
            if np.any(mse_list):
                metrics[label][model] = {'MSE_mean': np.mean(mse_list), 'MSE_stderr': sem(mse_list)}
                print(f'{model} {label} MSE: {np.mean(mse_list):.4f}')

    return metrics


def metrics_to_df(metrics, label):
    data = []
    for model in metrics[label]:
        if 'MAE_mean' in metrics[label][model]:
            data.append({'Model': model, 'Metric': 'MAE', 'Value': metrics[label][model]['MAE_mean'],
                         'Error': metrics[label][model]['MAE_stderr']})
            data.append({'Model': model, 'Metric': 'Correlation', 'Value': metrics[label][model]['Correlation_mean'],
                         'Error': metrics[label][model]['Correlation_stderr']})
        if 'Accuracy_mean' in metrics[label][model]:
            data.append({'Model': model, 'Metric': 'Accuracy', 'Value': metrics[label][model]['Accuracy_mean'],
                         'Error': metrics[label][model]['Accuracy_stderr']})
        if 'MSE_mean' in metrics[label][model]:
            data.append({'Model': model, 'Metric': 'MSE', 'Value': metrics[label][model]['MSE_mean'],
                         'Error': metrics[label][model]['MSE_stderr']})

    df = pd.DataFrame(data)
    return df
