"""Evaluate the unchanged collision kernel only around observed free centers."""
import cv2
import numpy as np


def observed_kernel_clear(known, kernel, inferred=None):
    """Return the original occupied-count<0.5/free-center boolean field.

    Every tested center retains a full kernel halo. Obstacles in that halo,
    including inferred obstacles, are included; outside-map padding stays zero.
    Cells that were not observed free remain false. No occupancy is changed.
    """
    known = np.asarray(known)
    kernel = np.asarray(kernel, dtype=np.float32)
    if kernel.ndim != 2 or any(size % 2 != 1 for size in kernel.shape):
        raise ValueError('Collision kernels require odd two-dimensional extents')
    free = known == 1
    rows = np.flatnonzero(free.any(axis=1))
    cols = np.flatnonzero(free.any(axis=0))
    clear = np.zeros(known.shape, dtype=bool)
    if not len(rows) or not len(cols):
        return clear
    ry, rx = kernel.shape[0] // 2, kernel.shape[1] // 2
    y0, y1 = max(0, rows[0]-ry), min(known.shape[0], rows[-1]+ry+1)
    x0, x1 = max(0, cols[0]-rx), min(known.shape[1], cols[-1]+rx+1)
    region = np.s_[y0:y1, x0:x1]
    blocked = known[region] == -1
    if inferred is not None:
        blocked |= inferred[region]
    counts = cv2.filter2D(blocked.astype(np.float32), -1, kernel,
                          borderType=cv2.BORDER_CONSTANT)
    clear[region] = (counts < .5) & free[region]
    return clear
