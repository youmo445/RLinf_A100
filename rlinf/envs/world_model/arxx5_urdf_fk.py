from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_URDF_PATH = "/data/zsq/x5.urdf"
DEFAULT_OFFSET_PATH = (
    "/data/zsq/RLinf/rlinf/envs/world_model/arxx5_fk_offsets.json"
)
DEFAULT_BASE_LINK = "base_link"
DEFAULT_TIP_LINK = "link6"


@dataclass
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis_xyz: np.ndarray


_CONFIG = {
    "urdf_path": DEFAULT_URDF_PATH,
    "offset_path": DEFAULT_OFFSET_PATH,
    "base_link": DEFAULT_BASE_LINK,
    "tip_link": DEFAULT_TIP_LINK,
    "offset_mode": "base",
    "state_unit": "rad",
}
_CHAIN_CACHE = None
_OFFSET_CACHE = None


def configure(
    urdf_path=None,
    offset_path=None,
    base_link=None,
    tip_link=None,
    offset_mode=None,
    state_unit=None,
):
    global _CHAIN_CACHE, _OFFSET_CACHE
    updates = {
        "urdf_path": urdf_path,
        "offset_path": offset_path,
        "base_link": base_link,
        "tip_link": tip_link,
        "offset_mode": offset_mode,
        "state_unit": state_unit,
    }
    changed_chain = False
    changed_offset = False
    for key, value in updates.items():
        if value is None:
            continue
        value = str(value)
        if _CONFIG.get(key) != value:
            _CONFIG[key] = value
            if key in ("urdf_path", "base_link", "tip_link"):
                changed_chain = True
            if key in ("offset_path", "offset_mode"):
                changed_offset = True
    if changed_chain:
        _CHAIN_CACHE = None
    if changed_offset:
        _OFFSET_CACHE = None


def _parse_xyz(text, default):
    if not text:
        return np.array(default, dtype=np.float64)
    values = [float(x) for x in text.split()]
    if len(values) != 3:
        raise ValueError(f"Expected 3 values, got {values}")
    return np.asarray(values, dtype=np.float64)


def _rpy_to_rot(rpy):
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _axis_angle_to_rot(axis, theta):
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c, s = math.cos(theta), math.sin(theta)
    C = 1.0 - c
    return np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ],
        dtype=np.float64,
    )


def _make_transform(rot, xyz):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3] = xyz
    return T


def _parse_urdf_chain(urdf_path, base_link, tip_link):
    root = ET.parse(urdf_path).getroot()
    joints = []
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue

        origin = joint.find("origin")
        axis = joint.find("axis")
        joints.append(
            JointSpec(
                name=joint.attrib["name"],
                joint_type=joint.attrib.get("type", "fixed"),
                parent=parent.attrib["link"],
                child=child.attrib["link"],
                origin_xyz=_parse_xyz(
                    origin.attrib.get("xyz") if origin is not None else None,
                    (0.0, 0.0, 0.0),
                ),
                origin_rpy=_parse_xyz(
                    origin.attrib.get("rpy") if origin is not None else None,
                    (0.0, 0.0, 0.0),
                ),
                axis_xyz=_parse_xyz(
                    axis.attrib.get("xyz") if axis is not None else None,
                    (1.0, 0.0, 0.0),
                ),
            )
        )

    child_to_joint = {joint.child: joint for joint in joints}
    chain_reversed = []
    current = tip_link
    while current != base_link:
        joint = child_to_joint.get(current)
        if joint is None:
            raise ValueError(
                f"Cannot trace URDF chain {base_link!r}->{tip_link!r}; stopped at {current!r}"
            )
        chain_reversed.append(joint)
        current = joint.parent
    return list(reversed(chain_reversed))


def _get_chain():
    global _CHAIN_CACHE
    if _CHAIN_CACHE is None:
        _CHAIN_CACHE = _parse_urdf_chain(
            Path(_CONFIG["urdf_path"]),
            _CONFIG["base_link"],
            _CONFIG["tip_link"],
        )
    return _CHAIN_CACHE


def _get_offsets():
    global _OFFSET_CACHE
    if _OFFSET_CACHE is not None:
        return _OFFSET_CACHE

    offsets = {
        "offset_mode": _CONFIG["offset_mode"],
        "left_offset_xyz": [-0.0951995117336301, -0.001000724923768371, -0.15649883982848967],
        "right_offset_xyz": [-0.09519835300344497, -0.000998287048561815, -0.15649921788495624],
    }
    path = Path(_CONFIG["offset_path"])
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        offsets.update(payload)

    if _CONFIG.get("offset_mode"):
        offsets["offset_mode"] = _CONFIG["offset_mode"]
    _OFFSET_CACHE = offsets
    return _OFFSET_CACHE


def _fk_transform(joint_values):
    T = np.eye(4, dtype=np.float64)
    q_idx = 0
    for joint in _get_chain():
        T = T @ _make_transform(_rpy_to_rot(joint.origin_rpy), joint.origin_xyz)
        if joint.joint_type in ("revolute", "continuous"):
            theta = float(joint_values[q_idx])
            q_idx += 1
            T = T @ _make_transform(
                _axis_angle_to_rot(joint.axis_xyz, theta),
                np.zeros(3, dtype=np.float64),
            )
    return T


def _rot_to_quat_wxyz(rot):
    trace = float(np.trace(rot))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rot[2, 1] - rot[1, 2]) / scale
        qy = (rot[0, 2] - rot[2, 0]) / scale
        qz = (rot[1, 0] - rot[0, 1]) / scale
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        scale = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        qw = (rot[2, 1] - rot[1, 2]) / scale
        qx = 0.25 * scale
        qy = (rot[0, 1] + rot[1, 0]) / scale
        qz = (rot[0, 2] + rot[2, 0]) / scale
    elif rot[1, 1] > rot[2, 2]:
        scale = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        qw = (rot[0, 2] - rot[2, 0]) / scale
        qx = (rot[0, 1] + rot[1, 0]) / scale
        qy = 0.25 * scale
        qz = (rot[1, 2] + rot[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        qw = (rot[1, 0] - rot[0, 1]) / scale
        qx = (rot[0, 2] + rot[2, 0]) / scale
        qy = (rot[1, 2] + rot[2, 1]) / scale
        qz = 0.25 * scale
    quat = np.array([qw, qx, qy, qz], dtype=np.float64)
    quat /= np.linalg.norm(quat) + 1e-12
    return quat


def _to_numpy(joints):
    if hasattr(joints, "detach"):
        joints = joints.detach().cpu().numpy()
    arr = np.asarray(joints, dtype=np.float64)
    if arr.shape[-1] != 6:
        raise ValueError(f"ARXX5 FK expects [..., 6] joints, got {arr.shape}")
    return arr


def _fk(joints, arm):
    joints_arr = _to_numpy(joints)
    single = joints_arr.ndim == 1
    joints_arr = joints_arr.reshape(-1, 6)
    if _CONFIG["state_unit"] == "deg":
        joints_arr = np.deg2rad(joints_arr)

    offsets = _get_offsets()
    offset = np.asarray(offsets[f"{arm}_offset_xyz"], dtype=np.float64)
    offset_mode = offsets.get("offset_mode", _CONFIG["offset_mode"])
    poses = []
    for row in joints_arr:
        T = _fk_transform(row)
        pos = T[:3, 3].copy()
        if offset_mode == "tool":
            pos = pos + T[:3, :3] @ offset
        elif offset_mode == "base":
            pos = pos + offset
        elif offset_mode not in ("none", None):
            raise ValueError(f"Unsupported FK offset mode: {offset_mode}")
        pose = np.zeros(7, dtype=np.float32)
        pose[:3] = pos.astype(np.float32)
        pose[3:] = _rot_to_quat_wxyz(T[:3, :3]).astype(np.float32)
        poses.append(pose)
    poses = np.stack(poses, axis=0)
    return poses[0] if single else poses


def left_fk(joints):
    return _fk(joints, "left")


def right_fk(joints):
    return _fk(joints, "right")
