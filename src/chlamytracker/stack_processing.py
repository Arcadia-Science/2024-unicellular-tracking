import warnings
from multiprocessing import Pool

import dask.array as da
import numpy as np
import skimage as ski

from swimtracker.utils import timeit


@timeit
def get_central_frames(stack, num_central_frames=100):
    """Crops `num_central_frames` from the center of an image stack.

    Parameters
    ----------
    stack : (Z, Y, X) array
        Input image stack of arbitrary dtype.
    num_central_frames : int
        Number of frames to crop from center.

    Notes
    -----
    * Cropped central frames are copied in memory so as not effect the original
      stack.
    """
    num_total_frames = stack.shape[0]

    if num_central_frames > num_total_frames:
        msg = (
            "Unable to crop central frames: Requested number of central "
            f"frames ({num_central_frames}) greater than number of "
            f"frames available ({num_total_frames})."
        )
        # raise ValueError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=1)
        return stack.copy()

    # indices of central slices
    z1, z2 = (
        num_total_frames // 2 - num_central_frames // 2,
        num_total_frames // 2 + num_central_frames // 2,
    )
    return stack[z1:z2].copy()


@timeit
def crop_out_roi(stack, center, radius):
    """Crops a square ROI out of an image stack along the first axis.

    Parameters
    ----------
    stack : (Z, Y, X) array
        Input image stack of arbitrary dtype.
    center : tuple[int, int]
        ROI center as an (x, y) coordinate.
    radius : float
        Radius to determine cropping window (1/2 width of square).

    Returns
    -------
    roi : (Z, Y, X) array
        Region of interest cropped from image stack with dimensions (Z, 2*R, 2*R).

    Raises
    ------
    IndexError
        If requested crop is outside the extent of the stack.
    """
    # validate input
    cx, cy = tuple(int(i) for i in center)
    r = round(radius)

    # crop to a rectangular roi
    nz, ny, nx = stack.shape
    y1, y2 = cy - r, cy + r
    x1, x2 = cx - r, cx + r
    if (y1 < 0) or (y2 > ny) or (x1 < 0) or (x2 > nx):
        msg = (
            f"Requested crop (array[:, {y1}:{y2}, {x1}:{x2}]) is out of bounds "
            f"for array with shape {stack.shape}."
        )
        raise IndexError(msg)
    else:
        roi = stack[:, y1:y2, x1:x2]

    return roi


@timeit
def rescale_to_float(stack):
    """Rescale intensity values of an image stack to (0, 1) range.

    Parameters
    ----------
    stack : (Z, Y, X) array
        Input image stack of arbitrary dtype.

    Notes
    -----
    * mimics `ski.exposure.rescale_intensity` but is slightly faster and less
      memory intensive at the cost of being less flexible.
    """
    _min = stack.min()
    _max = stack.max()
    rescaled = (stack - _min) / (_max - _min)
    return rescaled


@timeit
def otsu_threshold_3d(stack, num_central_frames=100):
    """Wrapper for `ski.filters.threshold_otsu` better equipped for handling
    large image stacks.

    Parameters
    ----------
    stack : (Z, Y, X) array
        Input image stack of arbitrary dtype.
    num_central_frames : int
        Number of central frames to use for determining the threshold. Useful
        for speeding up computation time when large stacks when
    """
    central_frames = get_central_frames(stack, num_central_frames)
    threshold = ski.filters.threshold_otsu(central_frames)
    return threshold


@timeit
def li_threshold_3d(stack, num_central_frames=100, initial_guess=0.1):
    """Wrapper for `ski.filters.threshold_li` better equipped for handling
    large image stacks.

    Parameters
    ----------
    stack : (Z, Y, X) array
        Input image stack of arbitrary dtype.
    num_central_frames : int
        Number of central frames to use for determining the threshold. Useful
        for speeding up computation time when large stacks when
    """
    central_frames = get_central_frames(stack, num_central_frames)
    threshold = ski.filters.threshold_li(central_frames, initial_guess=initial_guess)
    return threshold


def otsu_threshold_dask(dask_array):
    """Otsu thresholding function compatible with dask.

    This function is copied nearly verbatim from [1] except that numpy functions
    are swapped with their dask equivalent (`np` --> `da`).

    References
    ----------
    [1] https://github.com/scikit-image/scikit-image/blob/v0.22.0/skimage/filters/thresholding.py#L366-L379
    """
    # histogram
    bin_range = (dask_array.min(), dask_array.max())
    counts, bin_edges = da.histogram(dask_array, bins=256, range=bin_range)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    # class probabilities for all possible thresholds
    weight1 = da.cumsum(counts)
    weight2 = da.cumsum(counts[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = da.cumsum(counts * bin_centers) / weight1
    mean2 = (da.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = da.argmax(variance12)
    threshold = bin_centers[idx]
    return threshold


@timeit
def circular_alpha_mask(stack, sigma=1.6, num_workers=6):
    """Apply a circular alpha mask to every image in a stack along the first axis
    (in parallel).

    Transforms a rectangular prism to a cylinder. I.e. something like (but in 3D)

        [[1, 1, 1, 1, 1],      [[0, 0.5, 1, 0.5, 0],
         [1, 1, 1, 1, 1],       [0.5, 1, 1, 1, 0.5],
         [1, 1, 1, 1, 1],  -->  [  1, 1, 1, 1,   1],
         [1, 1, 1, 1, 1],       [0.5, 1, 1, 1, 0.5],
         [1, 1, 1, 1, 1]]       [0, 0.5, 1, 0.5, 0]]

    Parameters
    ----------
    stack : (Z, Y, X) array
        Input image stack of arbitrary dtype.
    sigma : float (optional)
        Standard deviation for Gaussian kernel.
    num_workers : int (optional)
        Number of processors to dedicate for multiprocessing.
    """
    # create alpha mask in the shape of a circle
    nz, ny, nx = stack.shape
    mask = np.zeros((nz, ny, nx))
    rr, cc = ski.draw.disk(
        center=(nx // 2, ny // 2),
        radius=(nx // 2),
    )
    mask[:, rr, cc] = 1
    # apply Guassian blur to mask
    mask_smoothed = gaussian_filter_3d_parallel(mask, sigma=sigma, num_workers=num_workers)

    # apply mask
    return stack * mask_smoothed


@timeit
def gaussian_filter_3d_parallel(stack, sigma=1.6, num_workers=6):
    """Apply a Gaussian filter to every image in a stack along the first axis
    (in parallel).

    Parameters
    ----------
    stack : (Z, Y, X) array
        Input image stack of arbitrary dtype.
    sigma : float (optional)
        Standard deviation for Gaussian kernel.
    num_workers : int (optional)
        Number of processors to dedicate for multiprocessing.
    """
    # list of sigmas for batch processing
    sigmas = [sigma] * stack.shape[0]
    # run Gaussian filter in parallel
    with Pool(num_workers) as workers:
        out = workers.starmap(ski.filters.gaussian, zip(stack, sigmas, strict=True))
    return np.array(out)


@timeit
def median_filter_3d_parellel(stack, radius=4, num_workers=6):
    """Apply a median filter to every image in a stack along the first axis
    (in parallel).

    While it is unfortunately much slower, a median filter is often useful
    because it is much better than a mean or Gaussian filter at preserving
    edges, which is desirable for cell detection and segmentation.

    Parameters
    ----------
    stack : (Z, Y, X) array
        Input image stack of arbitrary dtype.
    radius : int (optional)
        Radius of the disk used as the structuring element for the filter.
    num_workers : int (optional)
        Number of processors to dedicate for multiprocessing.

    Notes
    -----
    * Timing analysis showed diminishing returns beyond 6 workers.
    """
    # list of footprints for batch processing
    footprint = ski.morphology.disk(radius)
    footprints = [footprint] * stack.shape[0]
    # run median filter in parallel
    with Pool(num_workers) as workers:
        out = workers.starmap(ski.filters.median, zip(stack, footprints, strict=True))
    return np.array(out)


@timeit
def remove_small_objects_3d_parallel(stack, min_area=150, num_workers=6):
    """Remove small objects from a segmented image stack.

    Parameters
    ----------
    stack : (Z, Y, X) bool array
        Input segmented image stack of dtype `bool`.
    min_area : int (optional)
        The smallest allowable object size.
    num_workers : int (optional)
        Number of processors to dedicate for multiprocessing.

    Notes
    -----
    * Default value for `min_area` is derived from the area in px^2 of a 6um
      diameter object when imaging at 10x magnification.
    """
    # list of minimum areas for batch processing
    min_areas = [min_area] * stack.shape[0]
    # run remove small objects in parallel
    with Pool(num_workers) as workers:
        out = workers.starmap(
            ski.morphology.remove_small_objects, zip(stack, min_areas, strict=True)
        )
    return np.array(out)
