from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


ROBOTWIN_CAMERA_ALIASES = {
    "head": "head_camera",
    "main": "head_camera",
    "left_wrist": "left_camera",
    "right_wrist": "right_camera",
    "front": "front_camera",
}


@dataclass
class RobotWinCameraAnchor:
    source_path: str
    camera_names: List[str]
    intrinsic_by_cam: Dict[str, np.ndarray]
    w2c0_by_cam: Dict[str, np.ndarray]
    original_hw_by_cam: Dict[str, Tuple[int, int]]
    cam_to_ee_by_cam: Dict[str, np.ndarray]


def to_robotwin_camera_name(camera_name: str) -> str:
    return ROBOTWIN_CAMERA_ALIASES.get(str(camera_name), str(camera_name))


def _decode_rgb_shape(raw) -> Tuple[int, int]:
    if isinstance(raw, np.bytes_):
        raw = raw.tobytes()
    elif isinstance(raw, np.ndarray):
        raw = raw.tobytes()
    elif not isinstance(raw, (bytes, bytearray)):
        raw = bytes(raw)
    image = Image.open(io.BytesIO(raw))
    width, height = image.size
    return int(height), int(width)


def to_4x4_w2c(extrinsic_cv: np.ndarray) -> np.ndarray:
    extrinsic_cv = np.asarray(extrinsic_cv, dtype=np.float32)
    if extrinsic_cv.shape == (4, 4):
        return extrinsic_cv
    if extrinsic_cv.shape != (3, 4):
        raise ValueError(
            f"Expected RobotWin extrinsic_cv shape (3, 4) or (4, 4), got {extrinsic_cv.shape}"
        )
    out = np.eye(4, dtype=np.float32)
    out[:3, :] = extrinsic_cv
    return out


def scale_intrinsic(
    intrinsic: np.ndarray,
    in_h: int,
    in_w: int,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    intrinsic = np.asarray(intrinsic, dtype=np.float32).copy()
    intrinsic[0, 0] *= out_w / float(in_w)
    intrinsic[0, 2] *= out_w / float(in_w)
    intrinsic[1, 1] *= out_h / float(in_h)
    intrinsic[1, 2] *= out_h / float(in_h)
    return intrinsic


def pose_wxyz_to_transform(pose_wxyz: np.ndarray) -> np.ndarray:
    pose_wxyz = np.asarray(pose_wxyz, dtype=np.float64)
    if pose_wxyz.shape[-1] != 7:
        raise ValueError(f"Expected pose [..., 7], got {pose_wxyz.shape}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = pose_wxyz[:3]
    qw, qx, qy, qz = pose_wxyz[3:7]
    transform[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    return transform


def make_abs_actions_from_fk_wxyz(
    action14_seq: np.ndarray,
    left_pose_wxyz: np.ndarray,
    right_pose_wxyz: np.ndarray,
) -> np.ndarray:
    action14_seq = np.asarray(action14_seq, dtype=np.float32)
    left_pose_wxyz = np.asarray(left_pose_wxyz, dtype=np.float32)
    right_pose_wxyz = np.asarray(right_pose_wxyz, dtype=np.float32)
    if action14_seq.ndim != 2 or action14_seq.shape[-1] != 14:
        raise ValueError(f"Expected action14_seq [T, 14], got {action14_seq.shape}")
    if left_pose_wxyz.shape != (action14_seq.shape[0], 7):
        raise ValueError(
            f"Expected left_pose_wxyz {(action14_seq.shape[0], 7)}, got {left_pose_wxyz.shape}"
        )
    if right_pose_wxyz.shape != (action14_seq.shape[0], 7):
        raise ValueError(
            f"Expected right_pose_wxyz {(action14_seq.shape[0], 7)}, got {right_pose_wxyz.shape}"
        )

    out = np.zeros((action14_seq.shape[0], 16), dtype=np.float32)
    out[:, 0:3] = left_pose_wxyz[:, 0:3]
    out[:, 3:7] = left_pose_wxyz[:, [4, 5, 6, 3]]
    out[:, 7] = np.clip(action14_seq[:, 6], 0.0, 1.0) * 120.0
    out[:, 8:11] = right_pose_wxyz[:, 0:3]
    out[:, 11:15] = right_pose_wxyz[:, [4, 5, 6, 3]]
    out[:, 15] = np.clip(action14_seq[:, 13], 0.0, 1.0) * 120.0
    return out


class RobotWinCameraFKProjector:
    def __init__(self, camera_names: Sequence[str]):
        self.camera_names = [to_robotwin_camera_name(name) for name in camera_names]

    def build_anchor(
        self,
        source_path: str,
        left_pose0_wxyz: np.ndarray,
        right_pose0_wxyz: np.ndarray,
    ) -> RobotWinCameraAnchor:
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"RobotWin source HDF5 not found: {source}")

        left_T_w_e0 = pose_wxyz_to_transform(np.asarray(left_pose0_wxyz, dtype=np.float64))
        right_T_w_e0 = pose_wxyz_to_transform(np.asarray(right_pose0_wxyz, dtype=np.float64))

        intrinsic_by_cam: Dict[str, np.ndarray] = {}
        w2c0_by_cam: Dict[str, np.ndarray] = {}
        original_hw_by_cam: Dict[str, Tuple[int, int]] = {}
        cam_to_ee_by_cam: Dict[str, np.ndarray] = {}

        with h5py.File(source, "r") as f:
            for cam in self.camera_names:
                base = f"observation/{cam}"
                if base not in f:
                    raise KeyError(f"RobotWin camera {cam!r} not found in {source}")
                intrinsic_by_cam[cam] = np.asarray(
                    f[f"{base}/intrinsic_cv"][0], dtype=np.float32
                )
                w2c0 = to_4x4_w2c(f[f"{base}/extrinsic_cv"][0])
                w2c0_by_cam[cam] = w2c0
                original_hw_by_cam[cam] = _decode_rgb_shape(f[f"{base}/rgb"][0])

                if cam == "left_camera":
                    cam_to_ee_by_cam[cam] = (w2c0.astype(np.float64) @ left_T_w_e0).astype(
                        np.float32
                    )
                elif cam == "right_camera":
                    cam_to_ee_by_cam[cam] = (
                        w2c0.astype(np.float64) @ right_T_w_e0
                    ).astype(np.float32)

        return RobotWinCameraAnchor(
            source_path=str(source),
            camera_names=list(self.camera_names),
            intrinsic_by_cam=intrinsic_by_cam,
            w2c0_by_cam=w2c0_by_cam,
            original_hw_by_cam=original_hw_by_cam,
            cam_to_ee_by_cam=cam_to_ee_by_cam,
        )

    def build_w2c_sequences(
        self,
        anchor: RobotWinCameraAnchor,
        left_pose_wxyz: np.ndarray,
        right_pose_wxyz: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        left_pose_wxyz = np.asarray(left_pose_wxyz, dtype=np.float64)
        right_pose_wxyz = np.asarray(right_pose_wxyz, dtype=np.float64)
        num_frames = int(left_pose_wxyz.shape[0])
        if right_pose_wxyz.shape[0] != num_frames:
            raise ValueError(
                f"Left/right pose length mismatch: {left_pose_wxyz.shape}, {right_pose_wxyz.shape}"
            )

        out: Dict[str, np.ndarray] = {}
        for cam in self.camera_names:
            if cam == "head_camera" or cam == "front_camera":
                out[cam] = np.repeat(anchor.w2c0_by_cam[cam][None], num_frames, axis=0)
                continue

            if cam == "left_camera":
                poses = left_pose_wxyz
            elif cam == "right_camera":
                poses = right_pose_wxyz
            else:
                raise ValueError(f"Unsupported RobotWin camera for FK extrinsics: {cam}")

            if cam not in anchor.cam_to_ee_by_cam:
                raise KeyError(f"Missing RobotWin camera-to-EE anchor for camera {cam!r}")

            T_c_e = anchor.cam_to_ee_by_cam[cam].astype(np.float64)
            seq = []
            for pose in poses:
                T_w_e = pose_wxyz_to_transform(pose)
                seq.append(T_c_e @ np.linalg.inv(T_w_e))
            out[cam] = np.stack(seq, axis=0).astype(np.float32)
        return out
