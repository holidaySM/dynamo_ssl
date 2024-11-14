from typing import Dict

import numpy as np
import torch
import zarr
from datasets import Sequence, Value, Dataset, Features

from lerobot.image_utils import VideoFrameValue
from lerobot.utils import hf_transform_to_torch
from lerobot.video_utils import VideoFrame


def zarr_data_generator(zarr_dict: Dict[str, zarr.core.Array]):
    data_size = _check_sanity_and_get_data_size(zarr_dict)
    for i in range(data_size):
        record = {k: v[i] if not isinstance(v[i], np.ndarray) else torch.from_numpy(v[i]) for k, v in zarr_dict.items()}
        record['index'] = i
        yield record


def zarr_data_to_data_dict(zarr_dict: Dict[str, zarr.core.Array]):
    data_size = _check_sanity_and_get_data_size(zarr_dict)
    data_dict = {}
    for k, zarr_v in zarr_dict.items():
        np_array = zarr_v[:]
        np_array = torch.from_numpy(np_array) if not np_array.dtype == 'object' else np_array
        data_dict[k] = np_array
    data_dict['index'] = torch.from_numpy(np.arange(data_size))
    return data_dict


def _check_sanity_and_get_data_size(zarr_dict: zarr.Group):
    array_lengths = [v.shape[0] for v in zarr_dict.values()]
    for p, n in zip(array_lengths, array_lengths[1:]):
        assert p == n
    return array_lengths[0]


def to_hf_dataset(zarr_dict: Dict[str, zarr.core.Array]):
    features = {}

    features["observation.image"] = VideoFrame()
    features["observation.state"] = Sequence(
        length=zarr_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["action"] = Sequence(
        length=zarr_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["next.reward"] = Value(dtype="float32", id=None)
    features["next.done"] = Value(dtype="bool", id=None)
    features["next.success"] = Value(dtype="bool", id=None)
    features["index"] = Value(dtype="int64", id=None)

    # TODO: Need to adjust for very large datasets
    data_dict = zarr_data_to_data_dict(zarr_dict)
    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset


def read_zarr_data_to_data_dict(data_group: zarr.Group):
    data_size = _check_sanity_and_get_data_size(data_group)
    except_keys = ["index", "observation.image"]

    data_dict = {}
    for k, zarr_v in data_group.items():
        if k in except_keys:
            continue
        values = zarr_v[:]
        data_dict[k] = values

    data_dict['observation.image'] = np.array([
        {
            'episode_index': ep_idx,
            'timestamp': t,
            'frame_index': frame_idx
        } for ep_idx, t, frame_idx in zip(data_dict['episode_index'], data_dict['timestamp'], data_dict['frame_index'])
    ])
    data_dict['index'] = np.arange(data_size)
    return data_dict


def to_hf_dataset2(root_group: zarr.Group):
    data_group = root_group['data']

    features = {}

    features["observation.image"] = VideoFrameValue()
    features["observation.state"] = Sequence(
        length=data_group["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["action"] = Sequence(
        length=data_group["action"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["next.reward"] = Value(dtype="float32", id=None)
    features["next.done"] = Value(dtype="bool", id=None)
    features["next.success"] = Value(dtype="bool", id=None)
    features["index"] = Value(dtype="int64", id=None)

    data_dict = read_zarr_data_to_data_dict(data_group)
    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset


def read_data_from_zarr(zarr_group) -> Dict[str, zarr.core.Array]:
    zarr_dict = {k: v for k, v in zarr_group.items()}
    return zarr_dict
