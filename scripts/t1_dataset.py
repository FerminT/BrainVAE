from functools import partial
import nibabel as nib
import numpy as np
from torch import from_numpy, tensor
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from torchio import Compose, RandomSwap
from scripts.utils import num2vect, crop_center, position_encoding


class T1Dataset(Dataset):
    def __init__(self, input_shape, datapath, data, latent_dim, conditional_dim, age_range, invariant,
                 testing=False, transform=None):
        self.input_shape = input_shape
        self.datapath = datapath
        self.data = data
        self.transform = transform
        self.testing = testing
        self.age_mapping = age_mapping_function(conditional_dim, latent_dim, age_range, invariant)

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


def age_mapping_function(conditional_dim, latent_dim, age_range, invariant):
    num_bins = age_range[1] - age_range[0]
    if 1 < conditional_dim != num_bins:
        raise ValueError('conditional_dim does not match the bins/classes for the age range')
    if conditional_dim == 1:
        age_mapping = age_to_tensor
    elif invariant:
        encoding_matrix = position_encoding(num_ages=100, embed_dim=latent_dim)
        age_mapping = partial(sinusoidal_age, encoding_matrix=encoding_matrix)
    else:
        age_mapping = partial(soft_age, lower=age_range[0], upper=age_range[1], bin_step=1, bin_sigma=1)
    return age_mapping


def sinusoidal_age(age, encoding_matrix):
    return from_numpy(encoding_matrix[round(age)])


def soft_age(age, lower, upper, bin_step, bin_sigma):
    return from_numpy(num2vect(age, [lower, upper], bin_step, bin_sigma)[0])


def age_to_onehot(age, lower, num_classes):
    return one_hot(tensor(round(age) - lower), num_classes)


def age_to_tensor(age):
    return tensor(float(age)).unsqueeze(dim=0)


def transform(t1_img):
    return Compose([RandomSwap(p=0.5)])(t1_img)
