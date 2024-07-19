import logging
from functools import wraps
from time import time

import imageio
import numpy as np
import skimage as ski
from scipy import stats

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
    # if pixel intensity is > 1 then safe to assume it is not the border region
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


def map_p_value_to_asterisks(p_value):
    """Map P value to symbol representing statistical significance."""
    if p_value <= 0.0001:
        return "****"
    elif p_value <= 0.001:
        return "***"
    elif p_value <= 0.01:
        return "**"
    elif p_value <= 0.05:
        return "*"
    else:
        return "NS"


def annotate_statistical_significance(
    distribution_A,
    distribution_B,
    matplotlib_axis,
    min_sample_size=6,
    center_annotation=False,
):
    """Measure statistical significance of two distributions and annotate plot accordingly.

    Parameters
    ----------
    distribution_A, distribution_B : (N,) array-like
        Distributions for the statistical test.
    matplotlib_axis : `matplotlib.axes.Axes`
        Matplotlib axis to annotate.
    center_annotation : bool
        Whether to place the annotation in the center of the axis. Appropriate for when the x-axis
        is categorical as opposed to numerical.
    """
    if (distribution_A.size < min_sample_size) or (distribution_B.size < min_sample_size):
        msg = (
            "Sample size of one or both distributions less than min sample size: "
            f"{min_sample_size}."
        )
        raise ValueError(msg)

    # Mann-Whitney U test
    _, p_value = stats.mannwhitneyu(distribution_A, distribution_B, alternative="two-sided")
    # get appropriate number of asterisks based on p-value
    asterisks = map_p_value_to_asterisks(p_value)

    # sort out (x, y) coordinates for annotations
    if center_annotation:
        x_center = 0.5
        x_start, x_end = 0.3, 0.7
    else:
        x_limits = matplotlib_axis.get_xlim()
        x_range = x_limits[1] - x_limits[0]
        x_center = (np.median(distribution_A) + np.median(distribution_B)) / 2
        x_start = x_center - x_range / 8
        x_end = x_center + x_range / 8
    # y coordinate independent of plotting style
    y_limits = matplotlib_axis.get_ylim()
    y_center = 1.1 * y_limits[1]

    # plot annotation line and symbol illustrating stastistical significance
    matplotlib_axis.plot([x_start, x_end], [y_center, y_center], "k-")
    fontsize = 20 if asterisks != "NS" else 12
    matplotlib_axis.text(x_center, 1.05 * y_center, asterisks, fontsize=fontsize, ha="center")
