#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    image_0 = frame_sequence[0]
    size_default = 40

    def get_features(image):
        return cv2.goodFeaturesToTrack(image, maxCorners=100, qualityLevel=0.2, minDistance=size_default)

    points = get_features(image_0)
    n = len(points)
    ids = np.arange(n)

    corners = FrameCorners(
        ids=ids,
        points=np.array(points),
        sizes=np.array([size_default] * len(points))
    )

    builder.set_corners_at_frame(0, corners)

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        cur_points, status, err = cv2.calcOpticalFlowPyrLK((image_0 * 255.0).astype(np.uint8), (image_1 * 255.0).astype(np.uint8), points, None, **lk_params)
        status = status[:, 0]
        print((status.shape, cur_points.shape))

        cur_points = cur_points[status == 1, :, :]
        cur_ids = ids[status == 1]

        new_points = get_features(image_0)
        new_ids = n + np.arange(len(new_points))
        n += len(new_points)

        ids = np.concatenate((cur_ids, new_ids))
        points = np.concatenate((cur_points, new_points))

        corners = FrameCorners(
            ids=ids,
            points=np.array(points),
            sizes=np.array([size_default] * len(points))
        )

        builder.set_corners_at_frame(frame, corners)
        image_0 = image_1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter