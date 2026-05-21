# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import dataclasses
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override


from rlinf.models.embodiment.openpi.policies import arxx5bimanual_policy

@dataclasses.dataclass(frozen=True)
class LeRobotArxX5BimanualDataConfig(DataConfigFactory):
    default_prompt: str | None = None
    dataset_root: str | None = None
    obs_fields: tuple[str, ...] = ("state",)
    action_mode: Literal["state", "delta_state"] = "state"
    sides: tuple[str, ...] = ("left", "right")
    image_keys: tuple[str, ...] = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    dataset_image_keys: tuple[str, ...] | None = ("head", "left_wrist", "right_wrist")

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if self.action_mode == "delta_state" and (not self.obs_fields or self.obs_fields[0] != "state"):
            raise ValueError(f"action_mode={self.action_mode!r} requires obs_fields to start with 'state'.")

        dataset_image_keys = self.dataset_image_keys or self.image_keys
        if len(dataset_image_keys) != len(self.image_keys):
            raise ValueError(f"Expected dataset_image_keys to match image_keys, got {dataset_image_keys=}.")
        image_repack = {
            model_key: f"observation.images.{dataset_key}"
            for model_key, dataset_key in zip(self.image_keys, dataset_image_keys, strict=True)
        }
        obs_repack = {
            "state": {"state": "observation.state"},
            "delta_state": {"delta_state": "observation.delta_state"},
            "velocity": {"velocity": "observation.velocity"},
            "current": {"current": "observation.current"},
            "ee_pose": {"ee_pose": "observation.ee_pose"},
            "ee_delta_pose": {"ee_delta_pose": "observation.ee_delta_pose"},
        }
        state_repack = {}
        for field in self.obs_fields:
            state_repack.update(obs_repack[field])
        repack = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "image": image_repack,
                        **state_repack,
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[
                arxx5bimanual_policy.ArxX5BimanualInputs(
                    model_type=model_config.model_type,
                    sides=self.sides,
                    image_keys=self.image_keys,
                    obs_fields=self.obs_fields,
                )
            ],
            outputs=[arxx5bimanual_policy.ArxX5BimanualOutputs(sides=self.sides)],
        )
        if self.action_mode == "delta_state":
            # data_transforms = data_transforms.push(
            #     inputs=[
            #         _transforms.DeltaActions(_transforms.make_bool_mask(len(self.sides) * arxx5_policy.JOINTS))
            #     ]
            # )
            raise NotImplementedError("Do not support delta_state")
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        base_config = self.create_base_config(assets_dirs, model_config)
        return dataclasses.replace(
            base_config,
            dataset_root=self.dataset_root or base_config.dataset_root,
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )