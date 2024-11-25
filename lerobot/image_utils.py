import collections.abc
from pathlib import Path
from typing import List

import einops
import torch


def load_from_frame_torch(item: dict, frame_keys: List[str], frame_path: Path):
    for frame_key in frame_keys:
        frame_dicts = item[frame_key] if isinstance(item[frame_key], collections.abc.Iterable) else [item[frame_key]]
        episode_indices = [frame_dict['episode_index'] for frame_dict in frame_dicts]
        assert len(set(episode_indices)) == 1

        episode_index = episode_indices[0]
        frame_indices = [frame_dict['frame_index'] for frame_dict in frame_dicts]

        frames = torch.load(frame_path / f'episode_{episode_index}.pth')
        frames = einops.rearrange(frames[frame_indices], "T C H W -> T 1 C H W") / 255.0
        item[frame_key] = frames  # T V C H W
    return item
