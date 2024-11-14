from typing import Optional

import torch

from custom_datasets.core import TrajectoryDataset
from lerobot.episode_dataset import EpisodeDataset


class RolloutPushAnyDataset(TrajectoryDataset):
    def __init__(self, data_directory, subset_fraction: Optional[float] = None, relative=False):
        super().__init__()
        self._lerobot_dataset = EpisodeDataset.from_zarr(data_directory)

        self.relative = relative

        self.subset_fraction = subset_fraction if subset_fraction else 1.0
        self._num_episodes = int(self._lerobot_dataset.num_episodes * subset_fraction) if subset_fraction \
            else self._lerobot_dataset.num_episodes

    def get_seq_length(self, episode_idx):
        from_idx, to_idx = self._get_episode_from_to(episode_idx)
        return to_idx - from_idx

    def get_frames(self, episode_idx, frames):
        from_idx, to_idx = self._get_episode_from_to(episode_idx)
        episode_sequence = self._lerobot_dataset[from_idx:to_idx]

        obs = episode_sequence['observation.image'][frames].unsqueeze(1)
        act = torch.stack(episode_sequence['action'], dim=0)[frames]
        mask = torch.ones(len(act)).bool()
        return obs, act, mask

    def __getitem__(self, episode_idx):
        return self.get_frames(episode_idx, range(self.get_seq_length(episode_idx)))

    def __len__(self):
        return self._num_episodes

    def _get_episode_from_to(self, episode_idx):
        from_idx = self._lerobot_dataset.episode_data_index["from"][episode_idx].item()
        to_idx = self._lerobot_dataset.episode_data_index["to"][episode_idx].item()
        return from_idx, to_idx
