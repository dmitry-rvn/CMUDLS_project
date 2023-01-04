from typing import Optional, List

import numpy as np

from .autograd import Tensor


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
    """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(self, dataset: Dataset, batch_size: Optional[int] = 1, shuffle: bool = False, random_seed: int = None, device=None):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.device = device

        indices = np.arange(len(dataset))
        if self.shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        self.ordering = np.array_split(indices, range(batch_size, len(dataset), batch_size))

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.ordering):
            result = self.dataset[self.ordering[self.n]]
            self.n += 1
            if not self.device:
                return tuple(map(Tensor, result))
            return tuple(Tensor(r, device=self.device) for r in result)
        else:
            raise StopIteration


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
