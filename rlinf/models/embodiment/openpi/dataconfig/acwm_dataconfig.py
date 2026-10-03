# Copyright 2025 The RLinf Authors.
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

import dataclasses
import pathlib
from typing import Sequence

import jax
import numpy as np
import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.policies import aloha_policy
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override


@dataclasses.dataclass(frozen=True)
class DirectResizeImages(_transforms.DataTransformFn):
    """Backport the direct image resize used by the ACWM training checkout."""

    height: int
    width: int
    method: jax.image.ResizeMethod = jax.image.ResizeMethod.LINEAR

    def __call__(self, data):
        def resize(image):
            image = np.asarray(image)
            if image.shape[:2] == (self.height, self.width):
                return image
            resized = jax.image.resize(
                image,
                (self.height, self.width, image.shape[-1]),
                method=self.method,
            )
            if image.dtype == np.uint8:
                return np.round(np.asarray(resized)).clip(0, 255).astype(np.uint8)
            if np.issubdtype(image.dtype, np.floating):
                return np.asarray(resized, dtype=image.dtype)
            raise ValueError(f"Unsupported image dtype: {image.dtype}")

        data["image"] = {key: resize(value) for key, value in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class LeRobotACWMDataConfig(DataConfigFactory):
    """ACWM six-task PI0.5 transforms used by the 0710 full-finetune run."""

    action_sequence_keys: Sequence[str] = ("action",)
    adapt_to_pi: bool = False
    extra_delta_transform: bool = False
    resize_height: int = 224
    resize_width: int = 224

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        if self.extra_delta_transform:
            raise ValueError(
                "ACWM PI0.5 was trained with absolute actions; "
                "extra_delta_transform must remain false."
            )

        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.base_0",
                            "cam_left_wrist": "observation.images.left_wrist_0",
                            "cam_right_wrist": "observation.images.right_wrist_0",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[
                aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi),
                DirectResizeImages(self.resize_height, self.resize_width),
            ],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )
