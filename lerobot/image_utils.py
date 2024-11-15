import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, List, Union

import pyarrow as pa
import torch
import zarr
from datasets.features.features import register_feature


def load_from_frame_zarr(item: dict, frame_keys: List[str], frame_group: zarr.Group):
    for frame_key in frame_keys:
        frame_dicts = item[frame_key] if isinstance(item[frame_key], list) else [item[frame_key]]
        episode_indices = [frame_dict['episode_index'] for frame_dict in frame_dicts]
        assert len(set(episode_indices)) == 1

        episode_index = episode_indices[0]
        frame_indices = [frame_dict['frame_index'] for frame_dict in frame_dicts]

        frames = frame_group[f'episode_{episode_index}']['frames']
        selected_frame = frames[frame_indices]
        frame_tensor = torch.from_numpy(selected_frame).type(torch.float32).permute(0, 3, 1, 2) / 255
        item[frame_key] = frame_tensor  # T C H W
    return item


def load_from_frame_torch(item: dict, frame_keys: List[str], frame_path: Path):
    for frame_key in frame_keys:
        frame_dicts = item[frame_key] if isinstance(item[frame_key], list) else [item[frame_key]]
        episode_indices = [frame_dict['episode_index'] for frame_dict in frame_dicts]
        assert len(set(episode_indices)) == 1

        episode_index = episode_indices[0]
        frame_indices = [frame_dict['frame_index'] for frame_dict in frame_dicts]

        frames = torch.load(frame_path / f'episode_{episode_index}.pth')
        item[frame_key] = frames[frame_indices]  # T V C H W
    return item

@dataclass
class VideoFrameValue:
    """
    Provides a type for a dataset containing video frames.

    Example:

    ```python
    data_dict = [{"image": {"episode_index": 5, "timestamp": 0.3, "frame_index": 0}}]
    features = {"image": FrameValue()}
    Dataset.from_dict(data_dict, features=Features(features))
    ```
    """

    pa_type: ClassVar[Any] = pa.struct({
        "episode_index": pa.int64(),
        "timestamp": pa.float32(),
        "frame_index": pa.int64()
    })
    _type: str = field(default="VideoFrameValue", init=False, repr=False)

    def __call__(self):
        return self.pa_type


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        "'register_feature' is experimental and might be subject to breaking changes in the future.",
        category=UserWarning,
    )
    # to make VideoFrame available in HuggingFace `datasets`
    register_feature(VideoFrameValue, "VideoFrameValue")
