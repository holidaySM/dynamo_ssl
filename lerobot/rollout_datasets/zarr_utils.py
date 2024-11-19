import numpy as np
import zarr
from datasets import Sequence, Value, Dataset, Features

from lerobot.image_utils import VideoFrameValue
from lerobot.utils import hf_transform_to_torch


def _check_sanity_and_get_data_size(zarr_dict: zarr.Group):
    array_lengths = [v.shape[0] for v in zarr_dict.values()]
    for p, n in zip(array_lengths, array_lengths[1:]):
        assert p == n
    return array_lengths[0]


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


def to_hf_dataset(root_group: zarr.Group):
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
