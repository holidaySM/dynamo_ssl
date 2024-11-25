import os
import re
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset


def get_max_episode_id(episode_path):
    pattern = re.compile(r"episode_(\d+)\.pth")
    max_idx = -1
    for filename in os.listdir(episode_path):
        match = pattern.match(filename)
        if match:
            idx = int(match.group(1))
            max_idx = max(max_idx, idx)
    return max_idx


def episode_data_generator(episode_path):
    def gen():
        num_episodes = get_max_episode_id(episode_path) + 1
        total_frames = 0
        for episode_idx in range(num_episodes):
            d = torch.load(episode_path / f'episode_{episode_idx}.pth')
            episode_num_frames = len(d['frame_index'])
            d['index'] = torch.arange(total_frames, total_frames + episode_num_frames, 1)
            total_frames += episode_num_frames
            yield d

    return gen


def hf_transform_to_tensor_or_np(items_dict: dict):
    for key in items_dict:
        first_item = items_dict[key][0][0]
        if isinstance(first_item, str) or isinstance(first_item, dict):
            items_dict[key] = np.array(items_dict[key])
        else:
            items_dict[key] = torch.tensor(items_dict[key])
    return items_dict


def to_episode_hf_dataset(root_path: Path, transform=hf_transform_to_tensor_or_np):
    episode_path = root_path / 'episodes'
    dataset = Dataset.from_generator(generator=episode_data_generator(episode_path))
    dataset.set_transform(transform)
    return dataset
