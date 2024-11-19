from pathlib import Path
from typing import Optional

import torch

from custom_datasets.core import TrajectoryDataset
from lerobot.image_utils import load_from_frame_torch
from lerobot.memory_dataset import to_memory_dataset
from lerobot.utils import calculate_episode_data_index


class RolloutPushAnyMemDataset(TrajectoryDataset):
    def __init__(self, data_directory, subset_fraction: Optional[float] = None, relative=False):
        super().__init__()

        self.root_path = Path(data_directory)
        self.frame_path = self.root_path / "episode_frames"

        self._dataset = to_memory_dataset(self.root_path)
        self.episode_data_index = calculate_episode_data_index(self._dataset)
        self._hf_num_episodes = len(self._dataset['episode_index'].unique())

        self.relative = relative

        self.subset_fraction = subset_fraction if subset_fraction else 1.0
        self._num_episodes = int(self._hf_num_episodes * self.subset_fraction)

    def get_seq_length(self, episode_idx):
        from_idx, to_idx = self._get_episode_from_to(episode_idx)
        return to_idx - from_idx

    def get_frames(self, episode_idx, frame_indices):
        from_idx, _ = self._get_episode_from_to(episode_idx)
        frame_indices = [from_idx + i for i in frame_indices]
        items = self._dataset[frame_indices]

        items = load_from_frame_torch(items, ['observation.image'], self.frame_path)

        obs = items['observation.image']
        act = items['action']
        mask = torch.ones(len(act)).bool()
        return obs, act, mask

    def __getitem__(self, episode_idx):
        return self.get_frames(episode_idx, range(self.get_seq_length(episode_idx)))

    def __len__(self):
        return self._num_episodes

    def _get_episode_from_to(self, episode_idx):
        from_idx = self.episode_data_index["from"][episode_idx].item()
        to_idx = self.episode_data_index["to"][episode_idx].item()
        return from_idx, to_idx

    @property
    def states(self):
        return None

    @property
    def actions(self):
        return None
