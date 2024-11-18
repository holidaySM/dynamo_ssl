from typing import Union

import zarr
import torch
from torch.utils.data import Dataset
import numpy as np

from lerobot.rollout_datasets.zarr_utils import read_zarr_data_to_data_dict


def _convert_to_tensor(value: Union[np.ndarray, dict]):
    if not isinstance(value, np.ndarray):
        return value

    if value.dtype == 'object':
        return value
    return torch.from_numpy(value)


class MemoryDataset(Dataset):
    def __init__(self, data_dict):
        super().__init__()

        self._data_dict = data_dict
        self._transform = None

    def __getitem__(self, index):
        if isinstance(index, str):
            return self._transform(self._data_dict[index]) if self._transform else self._data_dict[index]

        d = {k: self._transform(v[index]) if self._transform else v[index] for k, v in self._data_dict.items()}
        return d

    def __len__(self):
        return len(self._data_dict['episode_index'])

    @classmethod
    def from_dict(cls, data_dict):
        return cls(data_dict)

    def set_transform(self, transform):
        self._transform = transform


def to_memory_dataset(root_group: zarr.Group):
    data_group = root_group['data']
    data_dict = read_zarr_data_to_data_dict(data_group)
    dataset = MemoryDataset.from_dict(data_dict)
    dataset.set_transform(_convert_to_tensor)
    return dataset
