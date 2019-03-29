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
import cv2
import sortednp as snp
from ba import run_bundle_adjustment

from pympler.tracker import SummaryTracker



def _find_triangulation(frame_corners_1, frame_corners_2, intrinsic_mat, triangulation_parameters):
    correspondences = build_correspondences(frame_corners_1, frame_corners_2)
    ids, points1, points2 = correspondences
    try:
        E, e_mask = cv2.findEssentialMat(points1, points2, intrinsic_mat, method=cv2.RANSAC)

        H, h_mask = cv2.findHomography(points1, points2,
                                      method=cv2.RANSAC, ransacReprojThreshold=1., confidence=0.999)
    except:
      return [], []

    if (E is None) or (e_mask is None) or (H is None) or (h_mask is None):
        return [], []

    mask = e_mask & h_mask
    if np.sum(mask) / np.sum(h_mask) < 0.7:
        return [], []

    try:
        r1, r2, t = cv2.decomposeEssentialMat(E)
    except:
        return [], []

    max_points_3d = []
    max_ids = []

    def to_pose(r, t):
        return Pose(np.linalg.inv(r), np.linalg.inv(r).dot(t))

    possible_poses = [to_pose(r1, t), to_pose(r1, -t), to_pose(r2, t), to_pose(r2, -t)]
    for v, pose in enumerate(possible_poses):
        view_mat_1 = eye3x4()
        view_mat_2 = pose_to_view_mat3x4(pose)
        points_3d, ids_3d = triangulate_correspondences(correspondences,
                                                        view_mat_1, view_mat_2, intrinsic_mat,
                                                        triangulation_parameters)
        if len(points_3d) > len(max_points_3d):
            max_points_3d = points_3d
            max_ids = ids_3d

    if len(max_ids) == 0:
        return [], []

    return max_points_3d, max_ids


def _initialize(corner_storage, intrinsic_mat, max_reprojection_error, min_depth):
    first_points = 100
    print(intrinsic_mat)
    n = len(corner_storage)
    assert n > 1
    angles = [16. / (1.5 ** i) for i in range(8)]

    point_clouds = []

    for angle in angles:
        triangulation_parameters = TriangulationParameters(
            max_reprojection_error=max_reprojection_error,
            min_triangulation_angle_deg=angle,
            min_depth=min_depth
        )
        max_points_3d = []
        max_ids = []

        for j in range(1, min(n, first_points)):
            points_3d, ids = _find_triangulation(corner_storage[0], corner_storage[j],
                                                 intrinsic_mat, triangulation_parameters)

            if len(points_3d) > len(max_points_3d):
                max_points_3d = points_3d
                max_ids = ids

        point_cloud = PointCloudBuilder()
        if len(max_points_3d) > 0:
            point_cloud.add_points(np.asarray(max_ids), np.asarray(max_points_3d))

        print("init: trying angle {}: {} points found".format(angle, len(max_points_3d)))

        point_clouds.append(point_cloud)

    max_size = np.max([len(cloud.points) for cloud in point_clouds if cloud.points is not None])

    for point_cloud, angle in zip(point_clouds, angles):
        if len(point_cloud.points) > max_size * 0.7:
            return point_cloud, angle

    return PointCloudBuilder(), 0.


def get_new_points(point_cloud, correspondences,
                   view_mat_1, view_mat_2,
                   intrinsic_mat, triangulation_parameters,
                   min_match):
    points_new, ids_new = triangulate_correspondences(correspondences,
                                                      view_mat_1, view_mat_2, intrinsic_mat,
                                                      triangulation_parameters)

    if len(ids_new) < min_match:
        return [], []

    ids_cloud = point_cloud.ids

    _, (indices_cloud, indices_in_cloud) = snp.intersect(ids_cloud.ravel(), ids_new.ravel(), indices=True)
    points_new = np.delete(points_new, indices_in_cloud, axis=0)
    ids_new = np.delete(ids_new, indices_in_cloud)

    if len(points_new) == 0:
        return [], []

    return np.asarray(ids_new), np.asarray(points_new)


def _track_camera(corner_storage: CornerStorage,
                  intrinsic_mat: np.ndarray) \
        -> Tuple[List[np.ndarray], PointCloudBuilder]:
    n = len(corner_storage)
    print(n)
    assert n > 1

    #memory_tracker = SummaryTracker()

    max_reprojection_error = 4.
    min_depth = 1.

    point_cloud, ok_angle = _initialize(corner_storage, intrinsic_mat, max_reprojection_error, min_depth)

    triangulation_parameters = TriangulationParameters(
        max_reprojection_error=max_reprojection_error / 2.,
        min_triangulation_angle_deg=ok_angle,
        min_depth=min_depth
    )

    start_size = len(point_cloud.points)

    views = [eye3x4()]
    invalid_ids = set()
    invalid_reprojection_error = 20
    for frame, frame_corners in enumerate(corner_storage[1:], start=1):
        if frame & (frame - 1) == 0: #frame is power two
            print("frame {}/{}".format(frame, len(corner_storage)))

        ids_points_2d = frame_corners.ids.flatten()
        ids_points_3d = point_cloud.ids.flatten()
        ids_common, (indices_points_2d, indices_points_3d) = snp.intersect(ids_points_2d, ids_points_3d, indices=True)

        points_3d = point_cloud.points[indices_points_3d]
        points_2d = frame_corners.points[indices_points_2d]

        valid_indices = ~np.isin(ids_common, invalid_ids)

        ids_common = ids_common[valid_indices]
        points_2d, points_3d = points_2d[valid_indices], points_3d[valid_indices]

        cur_view = eye3x4()
        if len(points_3d) > 7:
            pnp_res = cv2.solvePnPRansac(points_3d, points_2d, intrinsic_mat, np.array([]), flags=cv2.SOLVEPNP_EPNP)
            found, r, t, inliers = pnp_res
            cur_view = rodrigues_and_translation_to_view_mat3x4(r, t)

            reproj_errors = compute_reprojection_errors(points_3d, points_2d, intrinsic_mat @  cur_view)
            invalid_ids |= set(ids_common[reproj_errors > invalid_reprojection_error])

        views.append(cur_view)

        for frame_prev in range(max(frame-100, 0), frame, 2):
            correspondences = build_correspondences(corner_storage[frame_prev], corner_storage[frame])
            ids, _, _ = correspondences
            if len(ids) <= 3:
                continue
            cur_ids, cur_points = get_new_points(point_cloud,
                                                 correspondences,
                                                 views[frame_prev], views[frame],
                                                 intrinsic_mat, triangulation_parameters,
                                                 min_match=start_size * 0.9)
            if len(cur_points) > 0:
                print("{} new points, frames {} {}".format(len(cur_points), frame_prev, frame))
                point_cloud.add_points(cur_ids, cur_points)

    views = run_bundle_adjustment(intrinsic_mat, list(corner_storage), max_reprojection_error, views, point_cloud)
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
