from itertools import accumulate
from typing import List

import hydra

from custom_datasets.core import TrajectoryDataset


class ConcatDataset(TrajectoryDataset):
    def __init__(self, datasets: List[TrajectoryDataset]):
        super().__init__()
        self.datasets = datasets
        self._accumulated_sizes = list(accumulate([len(dataset) for dataset in datasets]))

    def get_seq_length(self, episode_idx):
        dataset, adjusted_episode_idx = self._get_dataset_and_adjusted_episode_idx(episode_idx)
        return dataset.get_seq_length(adjusted_episode_idx)

    def get_frames(self, episode_idx, frames):
        dataset, adjusted_episode_idx = self._get_dataset_and_adjusted_episode_idx(episode_idx)
        return dataset.get_frames(adjusted_episode_idx, frames)

    def __getitem__(self, episode_idx):
        dataset, adjusted_episode_idx = self._get_dataset_and_adjusted_episode_idx(episode_idx)
        return dataset[adjusted_episode_idx]

    def __len__(self):
        return self._accumulated_sizes[-1]

    def get_states(self, episode_idx, frame_idx=None):
        dataset, adjusted_episode_idx = self._get_dataset_and_adjusted_episode_idx(episode_idx)
        return dataset.get_states(adjusted_episode_idx, frame_idx)

    def get_actions(self, episode_idx, frame_idx=None):
        dataset, adjusted_episode_idx = self._get_dataset_and_adjusted_episode_idx(episode_idx)
        return dataset.get_actions(adjusted_episode_idx, frame_idx)

    def _get_dataset_and_adjusted_episode_idx(self, episode_idx):
        dataset_idx = self._get_dataset_idx(episode_idx)
        adjusted_episode_idx = episode_idx - self._get_episode_idx_offset(dataset_idx)
        return self.datasets[dataset_idx], adjusted_episode_idx

    def _get_dataset_idx(self, episode_idx):
        for dataset_idx, accumulated_size in enumerate(self._accumulated_sizes):
            if episode_idx < accumulated_size:
                return dataset_idx
        raise IndexError

    def _get_episode_idx_offset(self, dataset_idx):
        if dataset_idx == 0:
            return 0
        return self._accumulated_sizes[dataset_idx - 1]
