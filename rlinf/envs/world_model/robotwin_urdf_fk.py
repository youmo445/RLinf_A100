from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


DEFAULT_CONFIG_PATH = "/data/zsq/RoboTwin/assets/embodiments/aloha-agilex/config.yml"


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
    "config_path": DEFAULT_CONFIG_PATH,
    "state_unit": "rad",
}
_FK_CACHE = None


def configure(
    urdf_path=None,
    offset_path=None,
    base_link=None,
    tip_link=None,
    offset_mode=None,
    state_unit=None,
):
    """Configure RoboTwin FK.

    The world-model env passes cfg.fk_urdf_path as ``urdf_path``. For RoboTwin
    this is intentionally the embodiment ``config.yml`` path, because that file
    contains the URDF path, joint names, root pose and tool-frame transform.
    """
    del offset_path, base_link, tip_link, offset_mode
    global _FK_CACHE
    if urdf_path is not None and str(urdf_path) != _CONFIG["config_path"]:
        _CONFIG["config_path"] = str(urdf_path)
        _FK_CACHE = None
    if state_unit is not None:
        _CONFIG["state_unit"] = str(state_unit)


def _parse_xyz(text, default):
    if text is None:
        return np.asarray(default, dtype=np.float64)
    if isinstance(text, (list, tuple)):
        values = [float(x) for x in text]
    else:
        values = [float(x) for x in str(text).split()]
    if len(values) != 3:
        raise ValueError(f"Expected 3 values, got {values}")
    return np.asarray(values, dtype=np.float64)


def _rpy_to_rot(rpy):
    return Rotation.from_euler("xyz", rpy).as_matrix()


def _axis_angle_to_rot(axis, theta):
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    return Rotation.from_rotvec(axis / norm * float(theta)).as_matrix()


def _quat_wxyz_to_rot(quat):
    qw, qx, qy, qz = [float(x) for x in quat]
    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix()


def _rot_to_quat_wxyz(rot):
    quat_xyzw = Rotation.from_matrix(rot).as_quat()
    return np.asarray(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float64,
    )


def _make_transform(rot, xyz):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3] = xyz
    return T


class _UrdfSerialFK:
    def __init__(self, urdf_path: Path, root_pose):
        self.children_by_parent = {}
        self.child_links = set()
        robot = ET.parse(urdf_path).getroot()
        for elem in robot.findall("joint"):
            origin = elem.find("origin")
            axis = elem.find("axis")
            joint = JointSpec(
                name=elem.attrib["name"],
                joint_type=elem.attrib.get("type", "fixed"),
                parent=elem.find("parent").attrib["link"],
                child=elem.find("child").attrib["link"],
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
                    (0.0, 0.0, 1.0),
                ),
            )
            self.children_by_parent.setdefault(joint.parent, []).append(joint)
            self.child_links.add(joint.child)

        roots = sorted(set(self.children_by_parent) - self.child_links)
        if not roots:
            raise ValueError(f"Cannot infer URDF root link from {urdf_path}")
        self.root_link = roots[0]

        root_pose = np.asarray(root_pose, dtype=np.float64)
        self.root_transform = np.eye(4, dtype=np.float64)
        self.root_transform[:3, 3] = root_pose[:3]
        self.root_transform[:3, :3] = _quat_wxyz_to_rot(root_pose[3:7])

    def forward(self, joint_values):
        link_transforms = {self.root_link: self.root_transform}
        joint_transforms = {}
        stack = [self.root_link]
        while stack:
            parent = stack.pop()
            parent_tf = link_transforms[parent]
            for joint in self.children_by_parent.get(parent, []):
                value = float(joint_values.get(joint.name, 0.0))
                motion = np.eye(4, dtype=np.float64)
                if joint.joint_type in ("revolute", "continuous"):
                    motion[:3, :3] = _axis_angle_to_rot(joint.axis_xyz, value)
                elif joint.joint_type == "prismatic":
                    motion[:3, 3] = joint.axis_xyz * value
                child_tf = parent_tf @ _make_transform(_rpy_to_rot(joint.origin_rpy), joint.origin_xyz) @ motion
                joint_transforms[joint.name] = child_tf
                link_transforms[joint.child] = child_tf
                stack.append(joint.child)
        return joint_transforms


def _load_fk():
    global _FK_CACHE
    if _FK_CACHE is not None:
        return _FK_CACHE
    config_path = Path(_CONFIG["config_path"])
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    urdf_path = Path(cfg["urdf_path"])
    if not urdf_path.is_absolute():
        urdf_path = config_path.parent / urdf_path
    root_pose = cfg.get("robot_pose", [[0, 0, 0, 1, 0, 0, 0]])[0]
    _FK_CACHE = (cfg, _UrdfSerialFK(urdf_path, root_pose))
    return _FK_CACHE


def _transformed_eef_pose(cfg, joint_global_transform):
    global_trans_matrix = np.asarray(cfg["global_trans_matrix"], dtype=np.float64)
    delta_matrix = np.asarray(cfg["delta_matrix"], dtype=np.float64)
    dis = float(cfg.get("gripper_bias", 0.0)) - 0.12
    rot = joint_global_transform[:3, :3] @ global_trans_matrix @ delta_matrix
    pos = joint_global_transform[:3, 3] + rot @ np.asarray([dis, 0.0, 0.0])
    return np.concatenate([pos, _rot_to_quat_wxyz(rot)]).astype(np.float32)


def _to_numpy(joints):
    if hasattr(joints, "detach"):
        joints = joints.detach().cpu().numpy()
    arr = np.asarray(joints, dtype=np.float64)
    if arr.shape[-1] != 6:
        raise ValueError(f"RoboTwin FK expects [..., 6] joints, got {arr.shape}")
    return arr


def _fk(joints, arm_idx):
    joints_arr = _to_numpy(joints)
    single = joints_arr.ndim == 1
    joints_arr = joints_arr.reshape(-1, 6)
    if _CONFIG["state_unit"] == "deg":
        joints_arr = np.deg2rad(joints_arr)

    cfg, fk_robot = _load_fk()
    arm_joint_names = cfg["arm_joints_name"][arm_idx]
    ee_joint = cfg["ee_joints"][arm_idx]
    poses = []
    for row in joints_arr:
        joint_values = {name: float(value) for name, value in zip(arm_joint_names, row)}
        joint_transforms = fk_robot.forward(joint_values)
        if ee_joint not in joint_transforms:
            raise ValueError(f"RoboTwin FK cannot find ee joint {ee_joint!r}")
        poses.append(_transformed_eef_pose(cfg, joint_transforms[ee_joint]))
    poses = np.stack(poses, axis=0)
    return poses[0] if single else poses


def left_fk(joints):
    return _fk(joints, 0)


def right_fk(joints):
    return _fk(joints, 1)
