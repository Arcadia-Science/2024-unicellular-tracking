from multiprocessing import Pool

import numpy as np
import skimage as ski

from .utils import timeit


class MicrochamberPoolProcessor:
    """Class for processing timelapse microscopy data of an individual agar microchamber pool.

    TODO: detailed description of what processing steps this class seeks to accomplish.

    Parameters
    ----------
    stack : (T, Y, X) uint16 array
        Input timelapse microscopy image data of an agar microchamber pool.
    radius_median : scalar (optional)
        Radius of median filter to apply (preprocessing).
    sigma_gaussian : scalar (optional)
        Sigma of Gaussian filter for blurring the alpha mask (preprocessing).

    Attributes
    ----------
    is_preprocessed : bool
    is_segmented : bool
    stack_raw : (Z, Y, X) array
        Raw (unprocessed) image stack.
    stack_preprocessed : (Z, Y, X) array
        Pre-processed image stack (prior to segmentation).

    Methods
    -------
    preprocess()

    Notes
    -----
    * Default values for `radius_median` and `sigma_gaussian` were chosen
      empirically based on visual inspection of image data after preprocessing.

    References
    ----------
    [1] https://doi.org/10.57844/arcadia-v1bg-6b60
    """
    def __init__(
        self,
        stack,
        radius_median=4,
        sigma_gaussian=4,
    ):

        self.stack_raw = stack
        self.radius_median = radius_median
        self.sigma_gaussian = sigma_gaussian

        self.is_preprocessed = False
        self.is_segmented = False

    def has_cells(self, contrast_threshold=0.05):
        """Determine whether pool contains cells.

        Determination is based on the amount of contrast in the standard
        deviation projection, using the variance of intensity values as a proxy
        for contrast.

        TODO: more robust testing as this has only been tested on a small
              number of test images
        """
        # get dtype limits for normalization
        # (0, 65535) is expected but safer to check
        dtype_limit_max = max(ski.util.dtype_limits(self.stack_raw))

        # compute the standard deviation projection
        std_intensity_projection = self.stack_raw.std(axis=0)

        # use variance of intensity as measure of contrast
        normalized_contrast = std_intensity_projection.var() / dtype_limit_max

        return normalized_contrast > contrast_threshold

    @timeit
    def preprocess(self, remove_stationary_objects=True):
        """Preprocess pool for cell tracking.

        Process
        -------
        1) Median filter
        2) Invert contrast
        3) Blend with alpha mask
        4) Background subtraction

        Parameters
        ----------
        remove_stationary_objects : bool
            Whether to remove stationary objects from the pool (both junk
            and non-motile cells.)

        Notes
        -----
        * Alpha mask is created by applying a Gaussian blur to a circle with
          radius r = 1/2 width of stack.
        """

        # apply median filter
        pool_filtered = median_filter_3d_parellel(
            self.stack_raw,
            r_disk=self.radius_median
        )

        # invert contrast
        pool_inverted = ski.util.invert(
            pool_filtered
        )

        # create alpha mask in the shape of a circle
        nz, ny, nx = self.stack_raw.shape
        mask = np.zeros((nz, ny, nx))
        rr, cc = ski.draw.disk(
            center=(nx//2, ny//2),
            radius=(nx//2)
        )
        mask[:, rr, cc] = 1
        # apply Guassian blur to mask
        mask = ski.filters.gaussian(
            mask,
            sigma=self.sigma_gaussian
        )
        # apply mask (transforms rectangular prism --> tube)
        pool_tube = mask * pool_inverted

        # estimate background from a central column of intensity values
        dz = 100  # column height
        dy = round(60/100 * ny)  # column length = 60% total length of pool
        dx = round(60/100 * nx)  # column width = 60% total width of pool
        z1, z2 = nz//2 - dz//2, nz//2 + dz//2
        y1, y2 = ny//2 - dy//2, ny//2 + dy//2
        x1, x2 = nx//2 - dx//2, nx//2 + dx//2
        column = pool_tube[z1:z2, y1:y2, x1:x2]
        # subtract background by clipping at median intensity of central column
        pool_rescaled = ski.exposure.rescale_intensity(
            pool_tube,
            in_range=(np.median(column), pool_tube.max()),
            out_range=(0, 1)
        )

        # remove junk but also non-motile cells by subtracting the mean
        # intensity projection
        if remove_stationary_objects:
            pool_rescaled -= pool_rescaled.mean(axis=0)
            pool_rescaled = np.clip(
                pool_rescaled, 0, pool_rescaled.max()
            )

        self.is_preprocessed = True
        self.stack_preprocessed = pool_rescaled

    def segment(self):
        """Segment cells in preprocessed pool for tracking."""

        # if not self.is_preprocessed:
        #     self.preprocess()

        msg = "Robust segmentation is still in the works..."
        raise NotImplementedError(msg)


def median_filter_3d_parellel(stack, r_disk=4, n_workers=6):
    """Apply median filter to every image in a stack along the first axis
    (in parallel).

    While it is unfortunately much slower, a median filter is used here
    because it is much better than a mean or Gaussian filter at preserving
    edges, which is desirable for cell detection and segmentation.

    Parameters
    ----------
    r_disk : scalar
        Radius of morphological footprint for median filter.
    n_workers : int
        Number of processors to dedicate for multiprocessing.

    Notes
    -----
    * Timing analysis showed diminishing returns beyond 6 workers.
    """

    # make a bunch of footprints for batch processing
    footprint = ski.morphology.disk(r_disk)
    footprints = [footprint]*stack.shape[0]
    # run median filter in parallel
    with Pool(n_workers) as workers:
        out = workers.starmap(
            ski.filters.median, zip(stack, footprints, strict=False)
        )

    return np.array(out)
