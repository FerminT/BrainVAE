import pandas as pd
from os import cpu_count
from numpy import inf
from pandas import concat
from torch.utils.data import DataLoader
from torch import tensor
from pathlib import Path
from sklearn.model_selection import train_test_split
from scripts.t1_dataset import T1Dataset
from scripts import constants


def get_loader(dataset, batch_size, shuffle, num_workers=4):
    num_workers = min(cpu_count(), num_workers)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def load_metadata(datapath):
    metadata = pd.read_csv(datapath / constants.METADATA_PATH / f'{datapath.name}_image_baseline_metadata.csv')
    age_range = [int(metadata['age_at_scan'].min()), round(metadata['age_at_scan'].max() + 0.5)]
    bmi_range = [inf, -inf]
    if metadata['bmi'].notna().any():
        bmi_range = [int(metadata['bmi'].min()), round(metadata['bmi'].max() + 0.5)]
    return metadata, age_range, bmi_range


def combine_datasets(datasets, sample_size, val_size, test_size, splits_path, redo_splits, shuffle, random_state):
    train_datasets, val_datasets, test_datasets = [], [], []
    age_range, bmi_range = [inf, -inf], [inf, -inf]
    for dataset in datasets:
        metadata, dataset_age_range, dataset_bmi_range = load_metadata(dataset)
        train, val, test = load_splits(dataset, metadata, sample_size, val_size, test_size, splits_path, redo_splits,
                                       shuffle=shuffle, random_state=random_state)
        age_range = [min(age_range[0], dataset_age_range[0]), max(age_range[1], dataset_age_range[1])]
        bmi_range = [min(bmi_range[0], dataset_bmi_range[0]), max(bmi_range[1], dataset_bmi_range[1])]
        train_datasets.append(train)
        val_datasets.append(val)
        test_datasets.append(test)
    train, val, test = pd.concat(train_datasets), pd.concat(val_datasets), pd.concat(test_datasets)
    train, val, test = (shuffle_data(train, random_state), shuffle_data(val, random_state),
                        shuffle_data(test, random_state))
    return train, val, test, age_range, bmi_range


def load_datasets(dataset, input_shape, latent_dim, age_dim, sample_size,
                  val_size, test_size, splits_path, redo_splits, shuffle, random_state):
    datasets = get_datasets(dataset)
    datapath = Path(constants.DATA_PATH)
    train, val, test, age_range, bmi_range = combine_datasets(datasets, sample_size, val_size, test_size, splits_path,
                                                              redo_splits, shuffle, random_state)
    train_dataset = T1Dataset(input_shape, datapath, train, latent_dim, age_dim, age_range, bmi_range, testing=False)
    val_dataset = T1Dataset(input_shape, datapath, val, latent_dim, age_dim, age_range, bmi_range, testing=True)
    test_dataset = T1Dataset(input_shape, datapath, test, latent_dim, age_dim, age_range, bmi_range, testing=True)
    return train_dataset, val_dataset, test_dataset


def load_splits(datapath, metadata, sample_size, val_size, test_size, splits_path, redo, shuffle, random_state):
    train_csv, val_csv, test_csv = get_splits_files(datapath, splits_path)
    if train_csv.exists() and val_csv.exists() and test_csv.exists() and not redo:
        train = pd.read_csv(train_csv)
        val = pd.read_csv(val_csv)
        test = pd.read_csv(test_csv)
    else:
        train, val, test = generate_splits(metadata, val_size, test_size, shuffle, random_state)
        train_csv.absolute().parent.mkdir(parents=True, exist_ok=True)
        train.to_csv(train_csv, index=False)
        val.to_csv(val_csv, index=False)
        test.to_csv(test_csv, index=False)
    if datapath.name != 'ukbb' and sample_size:
        train = train.sample(sample_size, random_state=random_state, replace=True)
    return train, val, test


def generate_splits(data, val_size, test_size, shuffle, random_state):
    data = preprocess(data)
    train, val_test = train_test_split(data, test_size=val_size + test_size, shuffle=shuffle, random_state=random_state)
    val, test = train_test_split(val_test, test_size=test_size / (val_size + test_size), shuffle=shuffle,
                                 random_state=random_state)
    return train, val, test


def shuffle_data(data, random_state):
    return data.sample(frac=1, random_state=random_state)


def preprocess(data):
    data = data.drop_duplicates(subset='subject_id')
    return data


def get_splits_files(datapath, splits_path):
    splits_path = datapath / splits_path
    train_csv, val_csv, test_csv = splits_path / 'train.csv', splits_path / 'val.csv', splits_path / 'test.csv'
    return train_csv, val_csv, test_csv


def get_datasets(dataset):
    datapath = Path(constants.DATA_PATH)
    dataset_path = Path(dataset)
    if len(dataset_path.parts) == 1:
        datasets = [d for d in (datapath / dataset_path).iterdir() if d.is_dir()]
    else:
        datasets = [datapath / dataset_path]
    return datasets


def load_set(dataset, split, splits_path, random_state):
    datasets = get_datasets(dataset)
    train, val, test, age_range, bmi_range = combine_datasets(datasets, sample_size=None, val_size=None, test_size=None,
                                                              splits_path=splits_path, redo_splits=False, shuffle=True,
                                                              random_state=random_state)
    if split == 'val':
        data = val
    elif split == 'test':
        data = test
    else:
        data = train
    return data, age_range, bmi_range


def upsample_datasets(train, n_upsampled):
    for dataset in train['dataset'].unique():
        dataset_samples = train[train['dataset'] == dataset]
        n_samples = len(dataset_samples)
        if n_samples < n_upsampled:
            resampled = dataset_samples.sample(n=n_upsampled - n_samples, replace=True, random_state=42)
            train = concat([train, resampled])
    return train
