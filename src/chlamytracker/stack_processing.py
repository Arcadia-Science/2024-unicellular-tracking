from multiprocessing import Pool

import numpy as np
import skimage as ski

from chlamytracker.utils import timeit


@timeit
def get_central_frames(stack, num_central_frames=10):
    """Crops `num_central_frames` from the center of an image stack.

    Parameters
    ----------
    stack : ([Z, T], Y, X) array
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
            f"Requested number of central frames ({num_central_frames}) greater "
            f"than number of frames available ({num_total_frames})."
        )
        raise ValueError(msg)

    # indices of central slices
    z1, z2 = (
        num_total_frames // 2 - num_central_frames // 2,
        num_total_frames // 2 + num_central_frames // 2,
    )
    return stack[z1:z2].copy()


@timeit
def rescale_to_float(stack):
    """Rescale intensity values of an image stack to (0, 1) range.

    Parameters
    ----------
    stack : ([Z, T], Y, X) array
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
def otsu_threshold_3d(stack, num_central_frames=10):
    """Wrapper for `ski.filters.threshold_otsu` better equipped for handling
    large image stacks.

    Parameters
    ----------
    stack : ([Z, T], Y, X) array
        Input image stack of arbitrary dtype.
    num_central_frames : int
        Number of central frames to use for determining the threshold. Useful
        for speeding up computation time when large stacks when
    """
    central_frames = get_central_frames(stack, num_central_frames)
    threshold = ski.filters.threshold_otsu(central_frames)
    return threshold


@timeit
def gaussian_filter_3d_parallel(stack, sigma=1.6, num_workers=6):
    """Apply a Gaussian filter to every image in a stack along the first axis
    (in parallel).

    Parameters
    ----------
    stack : ([Z, T], Y, X) array
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
        out = workers.starmap(ski.filters.gaussian, zip(stack, sigmas, strict=False))
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
    stack : ([Z, T], Y, X) array
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
        out = workers.starmap(ski.filters.median, zip(stack, footprints, strict=False))
    return np.array(out)


@timeit
def remove_small_objects_3d_parallel(stack, min_area=150, num_workers=6):
    """Remove small objects from a segmented image stack.

    Parameters
    ----------
    stack : ([Z, T], Y, X) bool array
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
            ski.morphology.remove_small_objects, zip(stack, min_areas, strict=False)
        )
    return np.array(out)
