import nibabel as nib
import numpy as np
import pandas as pd
from torchio.transforms import RandomFlip
from os import cpu_count
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy, tensor
from sklearn.model_selection import train_test_split
from scripts.utils import num2vect


def get_loader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=cpu_count())


def load_datasets(datapath, input_shape, conditional_dim, sample_size, val_size, test_size, redo_splits,
                  shuffle, random_state):
    metadata = pd.read_csv(datapath / 'metadata' / f'{datapath.name}_image_baseline_metadata.csv')
    train, val, test = load_splits(datapath, metadata, sample_size, val_size, test_size, redo_splits,
                                   shuffle=shuffle, random_state=random_state)
    train_dataset = T1Dataset(input_shape, datapath, train, conditional_dim)
    val_dataset = T1Dataset(input_shape, datapath, val, conditional_dim)
    test_dataset = T1Dataset(input_shape, datapath, test, conditional_dim)
    return train_dataset, val_dataset, test_dataset


def load_splits(datapath, metadata, sample_size, val_size, test_size, redo, shuffle, random_state):
    splits_path = datapath / 'splits'
    if sample_size != -1:
        splits_path = splits_path / f'sample_{sample_size}'
    if splits_path.exists() and not redo:
        train = pd.read_csv(splits_path / 'train.csv')
        val = pd.read_csv(splits_path / 'val.csv')
        test = pd.read_csv(splits_path / 'test.csv')
    else:
        train, val, test = generate_splits(metadata, sample_size, val_size, test_size, shuffle, random_state)
        splits_path.mkdir(parents=True)
        train.to_csv(splits_path / 'train.csv', index=False)
        val.to_csv(splits_path / 'val.csv', index=False)
        test.to_csv(splits_path / 'test.csv', index=False)
    return train, val, test


def generate_splits(data, sample_size, val_size, test_size, shuffle, random_state):
    data = preprocess(data, sample_size)
    train, val_test = train_test_split(data, test_size=val_size + test_size, shuffle=shuffle, random_state=random_state)
    val, test = train_test_split(val_test, test_size=val_size / (val_size + test_size), shuffle=shuffle,
                                 random_state=random_state)
    return train, val, test


def preprocess(data, sample_size):
    data['age_at_scan'] = data['days_since_baseline'] / 365 + data['age_at_baseline']
    data = data.sort_values(['subject_id', 'days_since_baseline']).groupby('subject_id').first()
    if sample_size > 0:
        data = data.sample(n=sample_size, random_state=42)
    return data


def crop_center(data, shape):
    x, y, z = data.shape
    start_x = (x - shape[0]) // 2
    start_y = (y - shape[1]) // 2
    start_z = (z - shape[2]) // 2
    return data[start_x:-start_x, start_y:-start_y, start_z:-start_z]


class T1Dataset(Dataset):
    def __init__(self, input_shape, datapath, data, conditional_dim=0, transform=RandomFlip()):
        self.input_shape = input_shape
        self.datapath = datapath
        self.data = data
        self.transform = transform
        self.soft_label = conditional_dim > 0
        self.age_range = [int(data['age_at_scan'].min()), round(data['age_at_scan'].max() + 0.5)]
        if self.soft_label and (self.age_range[1] - self.age_range[0]) != conditional_dim:
            raise ValueError('conditional_dim should be equal to the number of bins in the age range')
        self.age_step = 1
        self.age_sigma = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        t1_img = self.load_and_process_img(sample)
        age = self.load_and_process_age(sample)
        return t1_img, age

    def load_and_process_img(self, sample):
        t1_img = nib.load(self.datapath / sample['image_path'])
        if self.transform:
            t1_img = self.transform(t1_img)
        t1_img = t1_img.get_fdata(dtype=np.float32)
        t1_img = t1_img / t1_img.mean()
        t1_img = crop_center(t1_img, self.input_shape)
        t1_img = from_numpy(np.asarray([t1_img]))
        return t1_img

    def load_and_process_age(self, sample):
        age = sample['age_at_scan']
        if not self.soft_label:
            age = tensor(float(age)).unsqueeze(dim=0)
        else:
            age, _ = num2vect(age, self.age_range, self.age_step, self.age_sigma)
            age = from_numpy(age)
        return age
