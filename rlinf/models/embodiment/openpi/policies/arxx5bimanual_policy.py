import dataclasses

import numpy as np

import openpi.models.model as _model
from openpi import transforms as _transforms

from .arxx5_policy import _make_state


SIDES = ("left", "right")
JOINTS_PER_SIDE = 7
MODEL_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def make_arxx5_example() -> dict:
    """Creates a random input example for the ArxX5 policy."""
    return {
        "image": {
            "base_0_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "left_wrist_0_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "right_wrist_0_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        },
        "state": np.random.rand(14).astype(np.float32),
        "prompt": "do something",
    }

def _to_uint8_hwc(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating):
        arr = (arr * 255).astype(np.uint8)
    return arr


@dataclasses.dataclass(frozen=True)
class ArxX5BimanualInputs(_transforms.DataTransformFn):
    model_type: _model.ModelType = _model.ModelType.PI05
    sides: tuple[str, ...] = SIDES
    image_keys: tuple[str, ...] = MODEL_IMAGE_KEYS
    obs_fields: tuple[str, ...] = ("state",)

    @property
    def action_dim(self) -> int:
        return len(self.sides) * JOINTS_PER_SIDE

    def __call__(self, data: dict) -> dict:
        unknown_sides = set(self.sides) - set(SIDES)
        if unknown_sides or not self.sides:
            raise ValueError(f"Expected sides to be a non-empty subset of {SIDES}, got {self.sides}.")
        unknown_image_keys = set(self.image_keys) - set(MODEL_IMAGE_KEYS)
        if unknown_image_keys or not self.image_keys:
            raise ValueError(
                f"Expected image_keys to be a non-empty subset of {MODEL_IMAGE_KEYS}, got {self.image_keys}."
            )

        in_images = data["image"]
        missing_images = set(self.image_keys) - set(in_images)
        if missing_images:
            raise ValueError(f"Missing arxx5 image keys: {sorted(missing_images)}")

        base_image = _to_uint8_hwc(in_images[self.image_keys[0]])
        images = {}
        image_masks = {}
        for key in MODEL_IMAGE_KEYS:
            if key in self.image_keys:
                images[key] = _to_uint8_hwc(in_images[key])
                image_masks[key] = np.True_
            else:
                images[key] = np.zeros_like(base_image)
                image_masks[key] = np.False_

        out = {
            "image": images,
            "image_mask": image_masks,
            "state": _make_state(data, self.obs_fields, self.action_dim),
        }

        if "actions" in data:
            out["actions"] = np.asarray(data["actions"], dtype=np.float32)[..., : self.action_dim]
        if "prompt" in data:
            out["prompt"] = data["prompt"]
        return out


@dataclasses.dataclass(frozen=True)
class ArxX5BimanualOutputs(_transforms.DataTransformFn):
    sides: tuple[str, ...] = SIDES

    @property
    def action_dim(self) -> int:
        return len(self.sides) * JOINTS_PER_SIDE

    def __call__(self, data: dict) -> dict:
        unknown_sides = set(self.sides) - set(SIDES)
        if unknown_sides or not self.sides:
            raise ValueError(f"Expected sides to be a non-empty subset of {SIDES}, got {self.sides}.")
        return {"actions": np.asarray(data["actions"], dtype=np.float32)[..., : self.action_dim]}
