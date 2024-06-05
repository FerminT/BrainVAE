import pandas as pd
from os import cpu_count
from pandas import read_csv
from torch.utils.data import DataLoader
from torch import tensor
from torch.nn.functional import one_hot
from sklearn.model_selection import train_test_split
from scripts.t1_dataset import T1Dataset
from scripts import constants


def get_loader(dataset, batch_size, shuffle, num_workers=4):
    num_workers = min(cpu_count(), num_workers)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def load_metadata(datapath):
    metadata = pd.read_csv(datapath / constants.METADATA_PATH / f'{datapath.name}_image_baseline_metadata.csv')
    age_range = [int(metadata['age_at_scan'].min()), round(metadata['age_at_scan'].max() + 0.5)]
    return metadata, age_range


def load_datasets(datapath, input_shape, conditional_dim, one_hot, sample_size, val_size, test_size, redo_splits,
                  shuffle, random_state):
    metadata, age_range = load_metadata(datapath)
    train, val, test = load_splits(datapath, metadata, sample_size, val_size, test_size, redo_splits,
                                   shuffle=shuffle, random_state=random_state)
    train_dataset = T1Dataset(input_shape, datapath, train, conditional_dim, age_range, one_hot, testing=False)
    val_dataset = T1Dataset(input_shape, datapath, val, conditional_dim, age_range, one_hot, testing=True)
    test_dataset = T1Dataset(input_shape, datapath, test, conditional_dim, age_range, one_hot, testing=True)
    return train_dataset, val_dataset, test_dataset


def load_splits(datapath, metadata, sample_size, val_size, test_size, redo, shuffle, random_state):
    train_csv, val_csv, test_csv = get_splits_files(datapath, sample_size)
    if train_csv.exists() and val_csv.exists() and test_csv.exists() and not redo:
        train = pd.read_csv(train_csv)
        val = pd.read_csv(val_csv)
        test = pd.read_csv(test_csv)
    else:
        train, val, test = generate_splits(metadata, sample_size, val_size, test_size, shuffle, random_state)
        train_csv.absolute().parent.mkdir(parents=True, exist_ok=True)
        train.to_csv(train_csv, index=False)
        val.to_csv(val_csv, index=False)
        test.to_csv(test_csv, index=False)
    return train, val, test


def generate_splits(data, sample_size, val_size, test_size, shuffle, random_state):
    data = preprocess(data)
    train, val_test = train_test_split(data, test_size=val_size + test_size, shuffle=shuffle, random_state=random_state)
    val, test = train_test_split(val_test, test_size=test_size / (val_size + test_size), shuffle=shuffle,
                                 random_state=random_state)
    if 0 < sample_size < len(data):
        train = train.sample(int(sample_size * (1 - val_size - test_size)), random_state=random_state)
        val = val.sample(int(sample_size * val_size), random_state=random_state)
        test = test.sample(int(sample_size * test_size), random_state=random_state)
    return train, val, test


def preprocess(data):
    data = data.drop_duplicates(subset='subject_id')
    return data


def gender_to_onehot(gender):
    label = 0 if gender == 'male' else 1
    return tensor(label).unsqueeze(dim=0)


def get_splits_files(datapath, sample_size):
    splits_path = datapath / constants.SPLITS_PATH
    if sample_size != -1:
        splits_path = splits_path / f'sample_{sample_size}'
    train_csv, val_csv, test_csv = splits_path / 'train.csv', splits_path / 'val.csv', splits_path / 'test.csv'
    return train_csv, val_csv, test_csv


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
