import numpy as np


def _as_batch(joints):
    joints = np.asarray(joints, dtype=np.float32)
    if joints.ndim == 1:
        joints = joints[None, :]
    if joints.shape[-1] != 6:
        raise ValueError(f"Expected joints [..., 6], got {joints.shape}")
    return joints


def left_fk(joints):
    joints = _as_batch(joints)
    pose = np.zeros((joints.shape[0], 7), dtype=np.float32)
    pose[:, 0] = -0.25 + 0.02 * joints[:, 0]
    pose[:, 1] = 0.15 + 0.02 * joints[:, 1]
    pose[:, 2] = 0.35 + 0.02 * joints[:, 2]
    pose[:, 3] = 1.0
    return pose


def right_fk(joints):
    joints = _as_batch(joints)
    pose = np.zeros((joints.shape[0], 7), dtype=np.float32)
    pose[:, 0] = 0.25 + 0.02 * joints[:, 0]
    pose[:, 1] = 0.15 + 0.02 * joints[:, 1]
    pose[:, 2] = 0.35 + 0.02 * joints[:, 2]
    pose[:, 3] = 1.0
    return pose


def fake_fk(joints):
    return left_fk(joints)
