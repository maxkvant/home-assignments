import cv2
from typing import List

import numpy as np

from corners import FrameCorners
from _camtrack import *
from scipy.optimize import approx_fprime as derivative
from collections import namedtuple
from collections import defaultdict
import sortednp as snp

__all__ = [
    'run_bundle_adjustment'
]


FrameParams = namedtuple(
    'ProjectionError',
    ('view_vec', 'ids_3d', 'points_2d')
)


def _view2vec(view):
    r_mat = view[:,:3]
    t = view[:,3]
    r, _ = cv2.Rodrigues(r_mat)
    return np.append(r, t)


def _vec2view(view_vec):
    r_vec = view_vec[0:3]
    t     = view_vec[3:6]
    view = rodrigues_and_translation_to_view_mat3x4(r_vec.reshape(3,1), t.reshape(3,1))
    return view


def _reprojection_errors(points3d: np.ndarray, points2d: np.ndarray,
                                proj_mat: np.ndarray) -> np.ndarray:
    projected_points = project_points(points3d, proj_mat)
    points2d_diff = projected_points - points2d
    return points2d_diff


def calculate_gradient(points_3d, points_2d, view_vec, intrinsic_mat):
    proj_mat = intrinsic_mat @ _vec2view(view_vec)
    n = len(points_3d)
    reproj_errors = _reprojection_errors(points_3d, points_2d, proj_mat) / n
    view_derivative = derivative(view_vec,
                                 lambda v: np.sum(_reprojection_errors(points_3d, points_2d, intrinsic_mat @ _vec2view(v)) ** 2) / n,
                                 np.full_like(view_vec, 1e-9)
                      )

    # f(x) = (Ax - b)^T * (Ax - b)
    # 1/2 * df / dx = A^T A x - A^T b = A^T * (Ax - b)
    reproj_errors = [np.append(reproj_error.reshape(-1), [0]) for reproj_error in reproj_errors]
    points_derivative = [2 * proj_mat.T.dot(reproj_error)[:3] for reproj_error in reproj_errors]
    return view_derivative, np.asarray(points_derivative)


def optimize_sgd(frames_params, point_cloud, intrinsic_mat, n_iters=100, alpha=1e-8):
    view_vecs = np.asarray([params.view_vec for params in frames_params])
    points_3d = point_cloud.points[:]


    def compute_errors(points_3d, view_vecs):
        errors = []
        for i, params in enumerate(frames_params):
            errors.append(np.average(
                compute_reprojection_errors(points_3d[params.ids_3d], params.points_2d, intrinsic_mat @ _vec2view(view_vecs[i]))
            ))
        return np.average(errors)

    current_best_error = compute_errors(points_3d, view_vecs)

    for iter in range(n_iters):
        gradient_views  = np.zeros(view_vecs.shape, view_vecs.dtype)
        gradient_points = np.zeros(points_3d.shape, points_3d.dtype)
        print(points_3d.shape)

        for i, params in enumerate(frames_params):
            cur_points_3d = points_3d[params.ids_3d]

            view_derivative, points_derivative = calculate_gradient(cur_points_3d, params.points_2d, view_vecs[i], intrinsic_mat)
            gradient_views[i] = view_derivative
            gradient_points[params.ids_3d] += points_derivative

            if i == 0:
                print(gradient_views[i])
                print(gradient_points[params.ids_3d][0])

        new_view_vecs = view_vecs - alpha * gradient_views
        new_points_3d = points_3d - alpha * gradient_points

        error_ave = compute_errors(new_points_3d, new_view_vecs)
        print("iter {} error_ave {}".format(iter, error_ave))

        if error_ave < current_best_error:
            current_best_error = error_ave

            view_vecs = new_view_vecs
            points_3d = new_points_3d
            alpha *= 1.2
        else:
            alpha /= 1.2

    point_cloud.update_points(point_cloud.ids.flatten(), points_3d)
    return view_vecs


def run_bundle_adjustment(intrinsic_mat: np.ndarray,
                          list_of_corners: List[FrameCorners],
                          max_inlier_reprojection_error: float,
                          view_mats: List[np.ndarray],
                          pc_builder: PointCloudBuilder) -> List[np.ndarray]:

    print("ba max_reproj={} {}".format(max_inlier_reprojection_error, len(pc_builder.ids)))
    n = len(list_of_corners)

    frames_params = []

    for frame in range(n):
        frame_corners = list_of_corners[frame]

        ids_points_2d = frame_corners.ids.flatten()
        ids_points_3d = pc_builder.ids.flatten()
        ids_common, (indices_points_2d, indices_points_3d) = snp.intersect(ids_points_2d, ids_points_3d, indices=True)
        points_3d = pc_builder.points[indices_points_3d]
        points_2d = frame_corners.points[indices_points_2d]

        reproj_errors = compute_reprojection_errors(points_3d, points_2d, intrinsic_mat @ view_mats[frame])
        indexes_valid = reproj_errors < max_inlier_reprojection_error

        frames_params.append(FrameParams(_view2vec(view_mats[frame]), indices_points_3d[indexes_valid], points_2d[indexes_valid]))

    view_vecs_adjusted = optimize_sgd(frames_params, pc_builder, intrinsic_mat)
    view_mats_adjusted = [_vec2view(view_vec) for view_vec in view_vecs_adjusted]
    return view_mats_adjusted



