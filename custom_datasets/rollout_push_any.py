from pathlib import Path
from typing import Optional

import torch

from custom_datasets.core import TrajectoryDataset
from lerobot.hf_dataset import to_episode_hf_dataset
from lerobot.image_utils import load_from_frame_torch


class RolloutPushAnyDataset(TrajectoryDataset):
    def __init__(self, data_directory, subset_fraction: Optional[float] = None, relative=False):
        super().__init__()

        self.root_path = Path(data_directory)
        self.frame_path = self.root_path / "episode_frames"

        self._episode_dataset = to_episode_hf_dataset(self.root_path)

        self.relative = relative

        self.subset_fraction = subset_fraction if subset_fraction else 1.0
        self._num_episodes = int(len(self._episode_dataset) * self.subset_fraction)

    def get_seq_length(self, episode_idx):
        return len(self._episode_dataset[episode_idx]['frame_index'])

    def get_frames(self, episode_idx, frame_indices):
        items = self._episode_dataset[episode_idx]
        items = {k: v[frame_indices] for k, v in items.items()}

        items = load_from_frame_torch(items, ['observation.image'], self.frame_path)

        obs = items['observation.image']
        act = items['action']
        mask = torch.ones(len(act)).bool()
        return obs, act, mask

    def __getitem__(self, episode_idx):
        return self.get_frames(episode_idx, range(self.get_seq_length(episode_idx)))

    def __len__(self):
        return self._num_episodes

    def get_states(self, episode_idx, frame_idx=None):
        if frame_idx is None:
            return self._episode_dataset[episode_idx]['observation.state']
        return self._episode_dataset[episode_idx]['observation.state'][frame_idx]

    def get_actions(self, episode_idx, frame_idx=None):
        if frame_idx is None:
            return self._episode_dataset[episode_idx]['action']
        return self._episode_dataset[episode_idx]['action'][frame_idx]
