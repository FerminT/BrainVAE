import nibabel as nib
import numpy as np
import pandas as pd
from torchio import Compose, RandomNoise, RandomFlip, RandomSwap
from os import cpu_count
from functools import partial
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy, tensor
from torch.nn.functional import one_hot
from sklearn.model_selection import train_test_split
from scripts.utils import num2vect, get_splits_files
from scripts import constants


def get_loader(dataset, batch_size, shuffle, num_workers):
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


def crop_center(data, shape):
    x, y, z = data.shape
    start_x = (x - shape[0]) // 2
    start_y = (y - shape[1]) // 2
    start_z = (z - shape[2]) // 2
    return data[start_x:-start_x, start_y:-start_y, start_z:-start_z]


def age_to_tensor(age):
    return tensor(float(age)).unsqueeze(dim=0)


def age_to_onehot(age, lower, num_classes):
    return one_hot(tensor(round(age) - lower), num_classes)


def soft_age(age, lower, upper, bin_step, bin_sigma):
    return from_numpy(num2vect(age, [lower, upper], bin_step, bin_sigma)[0])


def age_mapping_function(conditional_dim, age_range, one_hot_age):
    num_bins = age_range[1] - age_range[0]
    if (not one_hot_age and 1 < conditional_dim != num_bins) or (one_hot_age and num_bins + 1 != conditional_dim):
        raise ValueError('conditional_dim does not match the bins/classes for the age range')
    if conditional_dim <= 1:
        age_mapping = age_to_tensor
    elif one_hot_age:
        age_mapping = partial(age_to_onehot, lower=age_range[0], num_classes=conditional_dim)
    else:
        age_mapping = partial(soft_age, lower=age_range[0], upper=age_range[1], bin_step=1, bin_sigma=1)
    return age_mapping


def transform(t1_img):
    return Compose([RandomSwap(p=0.5)])(t1_img)


class EmbeddingDataset(Dataset):
    def __init__(self, data, target, transform_fn=None):
        if target not in data.columns:
            raise ValueError(f'{target} is not a column in the dataset')
        self.data = data
        self.target = target
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        target = sample[self.target]
        if self.transform_fn:
            target = self.transform_fn(target)
        return sample['embedding'], target


class T1Dataset(Dataset):
    def __init__(self, input_shape, datapath, data, conditional_dim, age_range, one_hot_age,
                 testing=False, transform=None):
        self.input_shape = input_shape
        self.datapath = datapath
        self.data = data
        self.transform = transform
        self.testing = testing
        self.age_mapping = age_mapping_function(conditional_dim, age_range, one_hot_age)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        t1_img, t1_transformed = self.load_and_process_img(sample)
        age = self.age_mapping(sample['age_at_scan'])
        return t1_img, t1_transformed, age

    def get_subject(self, subject_id):
        return self.data[self.data['subject_id'] == subject_id].iloc[0]

    def get_metadata(self, idx):
        return self.data.iloc[idx]

    def load_and_process_img(self, sample):
        t1_img = nib.load(self.datapath / sample['image_path'])
        t1_transformed = self.transform(t1_img) if self.transform and not self.testing else t1_img
        t1_img = self.preprocess_img(t1_img)
        t1_transformed = self.preprocess_img(t1_transformed)
        return t1_img, t1_transformed

    def preprocess_img(self, t1_img):
        t1_img = t1_img.get_fdata(dtype=np.float32)
        t1_img = t1_img / t1_img.mean()
        t1_img = crop_center(t1_img, self.input_shape)
        t1_img = from_numpy(np.asarray([t1_img]))
        return t1_img
