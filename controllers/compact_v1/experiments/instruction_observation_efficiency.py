"""Lossless observation storage and equivalent free-depth rasterization."""
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np


def free_ray_pixels(camera, ray_end, origin, resolution, shape):
    """Rasterize the existing samples without allocating a full XYZ volume.

    Preserve the original float64 linspace, arithmetic order, height limits,
    cell rounding and bounds. Separate coordinate arrays reduce temporary
    memory and avoid repeatedly reducing two-column bounds predicates.
    """
    delta = ray_end - camera
    alpha = np.linspace(0, 1, max(52, int(np.ceil(4. / resolution)) + 1))[:, None]
    z = camera[2] + alpha * delta[:, 2]
    keep = (z >= .10) & (z <= 1.65)
    columns = np.floor((camera[0] + alpha * delta[:, 0] - origin[0]) / resolution).astype(np.int64)
    rows = np.floor((camera[1] + alpha * delta[:, 1] - origin[1]) / resolution).astype(np.int64)
    keep &= (columns >= 0) & (columns < shape[1]) & (rows >= 0) & (rows < shape[0])
    free = np.zeros(shape, np.uint8)
    free[rows[keep], columns[keep]] = 1
    return free


def save_observation(path, fields):
    """Write standard, lossless NPZ with one buffered filesystem write.

    ZIP level one trades a modest increase in compressed bytes for less CPU
    time. Array values, dtypes, calibration and observation IDs are retained.
    """
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
        for key, value in fields.items():
            with archive.open(key + ".npy", "w", force_zip64=True) as member:
                np.lib.format.write_array(member, np.asanyarray(value), allow_pickle=False)
    path.write_bytes(buffer.getbuffer())
