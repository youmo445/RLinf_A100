from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw

from diffsynth.trainers.arxx5_dataset4_wancontrolmultiview import (
    ColorListLeft,
    ColorListRight,
    EndEffectorPts,
    compute_depth_radius,
    get_transformation_matrix_from_quat,
    grip_to_left_color,
    grip_to_right_color,
)


def _draw_alpha_ellipse(base, center_xy, radius, color, alpha):
    mask = Image.new("L", (base.shape[1], base.shape[0]), 0)
    draw = ImageDraw.Draw(mask)
    x, y = int(center_xy[0]), int(center_xy[1])
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=int(np.clip(alpha, 0, 255)),
    )
    foreground = Image.new(
        "RGB", (base.shape[1], base.shape[0]), tuple(int(value) for value in color)
    )
    return np.asarray(
        Image.composite(foreground, Image.fromarray(base), mask), dtype=np.uint8
    )


def generate_traj_map_alpha(
    abs_actions: np.ndarray,
    w2c: np.ndarray,
    intrinsic: np.ndarray,
    h: int,
    w: int,
    radius: int = 40,
    radius_mode: str = "perspective",
    min_radius: int = 14,
    max_radius: int = 58,
    ref_depth: float = 0.30,
    near_depth: float = 0.19,
    far_depth: float = 0.69,
    circle_alpha: int = 160,
    local_x_offset: float = 0.2,
    axis_line_width: int = 8,
) -> list[np.ndarray]:
    """Render the ACWM action map with alpha-blended gripper circles."""
    abs_actions = np.asarray(abs_actions, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    w2c = np.asarray(w2c, dtype=np.float32)
    if w2c.ndim == 2:
        w2c = np.repeat(w2c[None], abs_actions.shape[0], axis=0)

    ee_key_pts = EndEffectorPts.reshape(1, 4, 4).transpose(0, 2, 1)
    if axis_line_width <= 0:
        raise ValueError(f"axis_line_width must be positive, got {axis_line_width}")
    cvt_matrix = np.eye(4, dtype=np.float32)
    cvt_matrix[0, 3] = float(local_x_offset)
    cvt_matrix = cvt_matrix.reshape(1, 4, 4)
    pose_l_mat = get_transformation_matrix_from_quat(abs_actions[:, 0:7])
    pose_r_mat = get_transformation_matrix_from_quat(abs_actions[:, 8:15])
    ee2cam_l = np.matmul(np.matmul(w2c, pose_l_mat), cvt_matrix)
    ee2cam_r = np.matmul(np.matmul(w2c, pose_r_mat), cvt_matrix)
    pts_l = np.matmul(ee2cam_l, ee_key_pts)
    pts_r = np.matmul(ee2cam_r, ee_key_pts)

    intr = intrinsic.reshape(1, 3, 3)
    uvs_l = np.matmul(intr, pts_l[:, :3, :])
    uvs_l = (
        (uvs_l / (pts_l[:, 2:3, :] + 1e-8))[:, :2, :]
        .transpose(0, 2, 1)
        .astype(np.int64)
    )
    uvs_r = np.matmul(intr, pts_r[:, :3, :])
    uvs_r = (
        (uvs_r / (pts_r[:, 2:3, :] + 1e-8))[:, :2, :]
        .transpose(0, 2, 1)
        .astype(np.int64)
    )

    image_list = []
    for frame_idx in range(abs_actions.shape[0]):
        image = np.zeros((h, w, 3), dtype=np.uint8) + 50
        color_l = grip_to_left_color(abs_actions[frame_idx, 7].item() / 120.0)
        color_r = grip_to_right_color(abs_actions[frame_idx, 15].item() / 120.0)

        point_groups: Sequence = (uvs_l[frame_idx], uvs_r[frame_idx])
        pose_groups: Sequence = (pts_l[frame_idx], pts_r[frame_idx])
        for points, points_3d, color in zip(
            point_groups, pose_groups, (color_l, color_r), strict=True
        ):
            if float(points_3d[2, 0]) <= 1e-6:
                continue
            base = np.asarray(points[0])
            if base[0] < 0 or base[0] >= w or base[1] < 0 or base[1] >= h:
                continue
            radius_i = compute_depth_radius(
                float(points_3d[2, 0]),
                radius,
                radius_mode,
                min_radius,
                max_radius,
                ref_depth,
                near_depth,
                far_depth,
            )
            image = _draw_alpha_ellipse(
                image, base, radius_i, color, circle_alpha
            )

        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        for points, points_3d, colors in zip(
            point_groups,
            pose_groups,
            (ColorListLeft, ColorListRight),
            strict=True,
        ):
            if float(points_3d[2, 0]) <= 1e-6:
                continue
            base = np.asarray(points[0])
            if base[0] < 0 or base[0] >= w or base[1] < 0 or base[1] >= h:
                continue
            for point_idx, point in enumerate(points):
                if point_idx == 0 or float(points_3d[2, point_idx]) <= 1e-6:
                    continue
                draw.line(
                    [tuple(map(int, base)), tuple(map(int, point[:2]))],
                    fill=tuple(colors[point_idx - 1]),
                    width=axis_line_width,
                )
        image_list.append(np.asarray(pil_image, dtype=np.uint8))
    return image_list
