import dataclasses

import numpy as np

import openpi.models.model as _model
from openpi import transforms as _transforms


JOINTS = 7
MODEL_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
OBS_FIELDS = ("state", "delta_state", "velocity", "current", "ee_pose", "ee_delta_pose")


def _make_state(data: dict, obs_fields: tuple[str, ...], fallback_dim: int) -> np.ndarray:
    unknown_fields = set(obs_fields) - set(OBS_FIELDS)
    if unknown_fields:
        raise ValueError(f"Unknown ARX X5 observation fields: {sorted(unknown_fields)}")

    if not obs_fields:
        return np.zeros((0,), dtype=np.float32)
    
    if all(field in data for field in obs_fields):
        return np.concatenate([np.asarray(data[field], dtype=np.float32) for field in obs_fields], axis=-1)

    if obs_fields == ("state",) and "state" in data:
        return np.asarray(data["state"], dtype=np.float32)[..., :fallback_dim]

    missing = [field for field in obs_fields if field not in data]
    raise ValueError(f"Missing ARX X5 observation fields after repack: {missing}")


def _to_uint8_hwc(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating):
        arr = (arr * 255).astype(np.uint8)
    return arr


@dataclasses.dataclass(frozen=True)
class ArxX5Inputs(_transforms.DataTransformFn):
    model_type: _model.ModelType = _model.ModelType.PI05
    image_keys: tuple[str, ...] = ("base_0_rgb", "right_wrist_0_rgb")
    obs_fields: tuple[str, ...] = ("state",)

    def __call__(self, data: dict) -> dict:
        unknown_image_keys = set(self.image_keys) - set(MODEL_IMAGE_KEYS)
        if unknown_image_keys or not self.image_keys:
            raise ValueError(f"Expected image_keys to be a non-empty subset of {MODEL_IMAGE_KEYS}, got {self.image_keys}.")

        in_images = data["image"]
        missing_images = set(self.image_keys) - set(in_images)
        if missing_images:
            raise ValueError(f"Missing arxx5 image keys: {sorted(missing_images)}")

        base_image = _to_uint8_hwc(in_images["base_0_rgb"])
        images = {key: np.zeros_like(base_image) for key in MODEL_IMAGE_KEYS}
        image_masks = {key: np.False_ for key in MODEL_IMAGE_KEYS}

        images["base_0_rgb"] = base_image
        image_masks["base_0_rgb"] = np.True_
        for key in self.image_keys:
            if key == "base_0_rgb":
                continue
            images[key] = _to_uint8_hwc(in_images[key])
            image_masks[key] = np.True_

        out = {
            "image": images,
            "image_mask": image_masks,
            "state": _make_state(data, self.obs_fields, JOINTS),
        }

        if "actions" in data:
            out["actions"] = np.asarray(data["actions"], dtype=np.float32)[..., :JOINTS]
        if "prompt" in data:
            out["prompt"] = data["prompt"]
        return out


@dataclasses.dataclass(frozen=True)
class ArxX5Outputs(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"], dtype=np.float32)[..., :JOINTS]}
