import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


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


class T1Dataset(Dataset):
    def __init__(self, input_shape, datapath, data, transform=None):
        self.input_shape = input_shape
        self.datapath = datapath
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        t1_img = nib.load(self.datapath / sample['image_path'])
        t1_img = t1_img.get_fdata(dtype=np.float32)
        t1_img = t1_img / t1_img.mean()
        if self.transform:
            sample = self.transform(sample)
        # Crop center to input_shape
        return t1_img