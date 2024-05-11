import logging
from functools import wraps
from time import time

import imageio
import skimage as ski

logger = logging.getLogger(__name__)


def timeit(f):
    """Decorator for outputting the execution time of a function."""

    @wraps(f)
    def wrap(*args, **kwargs):
        t0 = time()
        result = f(*args, **kwargs)
        t1 = time()

        out = f"{f.__name__} :: {t1-t0:.2f}s"
        logger.info(out)
        return result

    return wrap


def configure_logger():
    """"""
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="[%(levelname)s][%(asctime)s] %(message)s",
        datefmt="%Y/%m/%d %I:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def crop_movie_to_content(filename, framerate):
    """Crop movie to content (remove borders).

    The napari canvas has a non-square default aspect ratio such that black
    borders appear on the side of the canvas when loading square image data.
    There does not appear to be a convenient way to set the canvas size via
    the napari API [1], so this function provides a way to brute-force remove
    the borders from a movie and overwrite it.

    References
    ----------
    [1] https://github.com/napari/napari/issues/4943
    """
    # load movie
    movie_data = ski.io.imread(filename)
    dimensions = movie_data.shape  # (Z, Y, X, C)

    # calculate dimensions of the video without border from first frame
    avg_y_intensity = movie_data[0].mean(axis=(1, 2))
    avg_x_intensity = movie_data[0].mean(axis=(0, 2))
    nonzero_height = avg_y_intensity[avg_y_intensity > 1].size
    nonzero_width = avg_x_intensity[avg_x_intensity > 1].size

    # calculate border dimensions
    border_y = (dimensions[1] - nonzero_height) // 2
    border_x = (dimensions[2] - nonzero_width) // 2

    # overwrite mp4 file
    writer = imageio.get_writer(
        filename,
        fps=framerate,
        quality=5,
        format="mp4",
    )

    # set x, y indices for cropping
    y1, y2 = border_y, border_y + nonzero_height
    x1, x2 = border_x, border_x + nonzero_width

    # create new movie with cropped frames
    for frame in movie_data:
        frame_cropped = frame[y1:y2, x1:x2]
        writer.append_data(frame_cropped)
    writer.close()
