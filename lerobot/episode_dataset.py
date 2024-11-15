#!/usr/bin/env python
from pathlib import Path
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, Optional, Dict, List

import datasets
import numpy as np
import torch.utils
import zarr

from lerobot.image_utils import load_from_frame_zarr, VideoFrameValue, load_from_frame_torch
from lerobot.rollout_datasets.zarr_utils import to_hf_dataset
from lerobot.utils import (
    calculate_episode_data_index,
    load_previous_and_future_frames,
)

CODEBASE_VERSION = "v1.6"


class EpisodeDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root_path: str,
            split: str = "train",
            image_transforms: Optional[Callable] = None,
            delta_timestamps: Optional[Dict[str, List[float]]] = None,
            info: Optional[dict] = None,
    ):
        super().__init__()
        self.root_path = Path(root_path)
        self.frame_path = self.root_path / "frames"

        self.root_group = zarr.open(root_path, mode='r')

        self.hf_dataset = to_hf_dataset(self.root_group)
        self.episode_data_index = calculate_episode_data_index(self.hf_dataset)

        self.split = split
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps

        self.info = info if info else {
            'fps': int(np.round(1 / self.hf_dataset[1]['timestamp'] - self.hf_dataset[0]['timestamp'])),
            'codevase_version': CODEBASE_VERSION,
        }

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.info["fps"]

    @property
    def features(self) -> datasets.Features:
        return self.hf_dataset.features

    @property
    def camera_keys(self) -> List[str]:
        """Keys to access image and video stream from cameras."""
        keys = []
        for key, feats in self.hf_dataset.features.items():
            if isinstance(feats, (datasets.Image, VideoFrameValue)):
                keys.append(key)
        return keys

    @property
    def video_frame_keys(self) -> List[str]:
        """Keys to access video frames that requires to be decoded into images.

        Note: It is empty if the dataset contains images only,
        or equal to `self.cameras` if the dataset contains videos only,
        or can even be a subset of `self.cameras` in a case of a mixed image/video dataset.
        """
        video_frame_keys = []
        for key, feats in self.hf_dataset.features.items():
            if isinstance(feats, VideoFrameValue):
                video_frame_keys.append(key)
        return video_frame_keys

    @property
    def num_samples(self) -> int:
        """Number of samples/frames."""
        return len(self.hf_dataset)

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return len(self.hf_dataset.unique("episode_index"))

    @property
    def tolerance_s(self) -> float:
        """Tolerance in seconds used to discard loaded frames when their timestamps
        are not close enough from the requested frames. It is only used when `delta_timestamps`
        is provided or when loading video frames from mp4 files.
        """
        # 1e-4 to account for possible numerical error
        return 1 / self.fps - 1e-4

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]

        if self.delta_timestamps is not None:
            item = load_previous_and_future_frames(
                item,
                self.hf_dataset,
                self.episode_data_index,
                self.delta_timestamps,
                self.tolerance_s,
            )

        item = load_from_frame_torch(item, self.video_frame_keys, self.frame_path)

        if self.image_transforms is not None:
            for cam in self.camera_keys:
                item[cam] = self.image_transforms(item[cam])

        return item

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository ID: '{self.repo_id}',\n"
            f"  Split: '{self.split}',\n"
            f"  Number of Samples: {self.num_samples},\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Recorded Frames per Second: {self.fps},\n"
            f"  Camera Keys: {self.camera_keys},\n"
            f"  Video Frame Keys: {self.video_frame_keys if self.video_frame_keys else 'N/A'},\n"
            f"  Transformations: {self.image_transforms},\n"
            f"  Codebase Version: {self.info.get('codebase_version', '< v1.6')},\n"
            f")"
        )

    @classmethod
    def from_zarr(cls,
                  root_path: str,
                  split: str = "train",
                  transform: callable = None,
                  delta_timestamps: Optional[Dict[str, List[float]]] = None):
        return EpisodeDataset(
            root_path,
            split,
            transform,
            delta_timestamps
        )
