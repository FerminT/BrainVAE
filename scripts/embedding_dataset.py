from torch import from_numpy
from torch.utils.data import Dataset


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
        embedding = from_numpy(sample['embedding'])
        target = sample[self.target]
        if self.transform_fn:
            target = self.transform_fn(target)
        return embedding, target
