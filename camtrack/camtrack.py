#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Tuple

import numpy as np

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import *
import math
import cv2
import sortednp as snp


def rotation_matrix2euler(R):
    def is_rot_matrix(R):
        Rt = np.transpose(R)
        should_be_identity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - should_be_identity)
        return n < 1e-6

    assert is_rot_matrix(R)

    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def _find_triangulation(frame_corners_1, frame_corners_2, intrinsic_mat, triangulation_parameters):
    correspondences = build_correspondences(frame_corners_1, frame_corners_2)
    ids, points1, points2 = correspondences

    # TODO check matrix E
    # H, h_mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    E, e_mask = cv2.findEssentialMat(points1, points2, intrinsic_mat, method=cv2.RANSAC)
    r1, r2, t = cv2.decomposeEssentialMat(E)

    max_points_3d = []
    max_ids = []

    possible_poses = [Pose(r1, t), Pose(r1, -t), Pose(r2, t), Pose(r2, -t)]
    for v, pose in enumerate(possible_poses):
        view_mat_1 = eye3x4()
        view_mat_2 = pose_to_view_mat3x4(pose)
        points_3d, ids_3d = triangulate_correspondences(correspondences,
                                                        view_mat_1, view_mat_2, intrinsic_mat,
                                                        triangulation_parameters)
        if len(points_3d) > len(max_points_3d):
            max_points_3d = points_3d
            max_ids = ids_3d
    return max_points_3d, max_ids


def _initialize(corner_storage, intrinsic_mat):
    print(intrinsic_mat)
    n = len(corner_storage)
    assert n > 1
    angles = [16. / (1.2 ** i) for i in range(20)]

    point_clouds = []

    for angle in angles:
        triangulation_parameters = TriangulationParameters(
            max_reprojection_error=2.,
            min_triangulation_angle_deg=angle,
            min_depth=.5
        )
        max_points_3d = []
        max_ids = []

        for j in range(1, n):
            points_3d, ids = _find_triangulation(corner_storage[0], corner_storage[j],
                                                 intrinsic_mat, triangulation_parameters)

            if len(points_3d) > len(max_points_3d):
                max_points_3d = points_3d
                max_ids = ids

        point_cloud = PointCloudBuilder()
        point_cloud.add_points(np.asarray(max_ids), np.asarray(max_points_3d))

        print("init trying angle {}: {} points found", angle, len(max_points_3d))

        point_clouds.append(point_cloud)

    max_size = np.max([len(cloud.points) for cloud in point_clouds])

    for point_cloud in point_clouds:
        if len(point_cloud.points) > max_size * 0.7:
            return point_cloud

    return PointCloudBuilder()


def _track_camera(corner_storage: CornerStorage,
                  intrinsic_mat: np.ndarray) \
        -> Tuple[List[np.ndarray], PointCloudBuilder]:
    # TODO: implement
    n = len(corner_storage)
    print(n)
    assert n > 1

    point_cloud = _initialize(corner_storage, intrinsic_mat)

    views = [eye3x4()]
    for frame_corners in corner_storage[1:]:
        ids_points_2d = frame_corners.ids.flatten()
        ids_points_3d = point_cloud.ids.flatten()
        _, (indices_points_2d, indices_points_3d) = snp.intersect(ids_points_2d, ids_points_3d, indices=True)

        points_3d = point_cloud.points[indices_points_3d]
        points_2d = frame_corners.points[indices_points_2d]

        pnp_res = cv2.solvePnPRansac(points_3d, points_2d, intrinsic_mat, np.array([]), flags=cv2.SOLVEPNP_EPNP)
        found, r, t, inliers = pnp_res
        cur_view = rodrigues_and_translation_to_view_mat3x4(r, t)
        views.append(cur_view)

    return views, point_cloud


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    view_mats, point_cloud_builder = _track_camera(
        corner_storage,
        intrinsic_mat
    )
    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    create_cli(track_and_calc_colors)()
