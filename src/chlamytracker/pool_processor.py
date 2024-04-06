from multiprocessing import Pool

import numpy as np
import skimage as ski

from .stack_processing import median_filter_3d_parellel
from .utils import timeit


class PoolSegmenter:
    """Class for processing timelapse microscopy data of an individual agar microchamber pool.

    TODO: detailed description of what processing steps this class seeks to accomplish.

    Parameters
    ----------
    stack : (T, Y, X) uint16 array
        Input timelapse microscopy image data of an individual agar microchamber
        pool that has been tightly cropped to either manually or e.g. after
        being detected with `PoolFinder.find_pools()`.
    median_filter_radius : scalar (optional)
        Radius of median filter to apply (preprocessing).
    gaussian_filter_sigma : scalar (optional)
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
    segment()

    Notes
    -----
    * Default values for `median_filter_radius` and `gaussian_filter_sigma`
      were chosen empirically based on visual inspection of image data after
      preprocessing.

    References
    ----------
    [1] https://doi.org/10.57844/arcadia-v1bg-6b60
    """

    def __init__(
        self,
        stack,
        median_filter_radius=4,
        gaussian_filter_sigma=4,
    ):
        self.stack_raw = stack
        self.median_filter_radius = median_filter_radius
        self.gaussian_filter_sigma = gaussian_filter_sigma

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

        TODO: these preprocessing steps are applicable only for brightfield
              microscopy datasets (inverting the contrast in particular), and
              will have to be adjusted for other imaging modalities.

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
        pool_filtered = median_filter_3d_parellel(self.stack_raw, r_disk=self.median_filter_radius)

        # invert contrast
        pool_inverted = ski.util.invert(pool_filtered)

        # create alpha mask in the shape of a circle
        nz, ny, nx = self.stack_raw.shape
        mask = np.zeros((nz, ny, nx))
        rr, cc = ski.draw.disk(center=(nx // 2, ny // 2), radius=(nx // 2))
        mask[:, rr, cc] = 1
        # apply Guassian blur to mask
        mask = ski.filters.gaussian(mask, sigma=self.gaussian_filter_sigma)
        # apply mask (transforms rectangular prism --> tube)
        pool_tube = mask * pool_inverted

        # estimate background from a central column of intensity values
        dz = 100  # column height
        dy = round(60 / 100 * ny)  # column length = 60% total length of pool
        dx = round(60 / 100 * nx)  # column width = 60% total width of pool
        z1, z2 = nz // 2 - dz // 2, nz // 2 + dz // 2
        y1, y2 = ny // 2 - dy // 2, ny // 2 + dy // 2
        x1, x2 = nx // 2 - dx // 2, nx // 2 + dx // 2
        column = pool_tube[z1:z2, y1:y2, x1:x2]
        # subtract background by clipping at median intensity of central column
        pool_rescaled = ski.exposure.rescale_intensity(
            pool_tube, in_range=(np.median(column), pool_tube.max()), out_range=(0, 1)
        )

        # remove junk but also non-motile cells by subtracting the mean
        # intensity projection
        if remove_stationary_objects:
            pool_rescaled -= pool_rescaled.mean(axis=0)
            pool_rescaled = np.clip(pool_rescaled, 0, pool_rescaled.max())

        self.is_preprocessed = True
        self.stack_preprocessed = pool_rescaled

    @timeit
    def segment(self, li_threshold_guess=0.1, min_object_size=500, filled_volume_threshold=0.1):
        """Segment cells in preprocessed pool for tracking."""

        if not self.is_preprocessed:
            self.preprocess()

        # reject noisy / low signal preprocessed stacks
        if li_threshold_guess > self.stack_preprocessed.max():
            msg = (
                "Insufficient signal strength for segmentation: max intensity "
                f"{self.stack_preprocessed.max():.2f} below threshold "
                f"{li_threshold_guess:.2f}."
            )
            raise ValueError(msg)

        # TODO: justify Li thresholding and initial guess
        threshold = ski.filters.threshold_li(
            self.stack_preprocessed, initial_guess=li_threshold_guess
        )
        stack_segmented = self.stack_preprocessed > threshold

        # filter out small objects
        stack_segmented = ski.morphology.remove_small_objects(stack_segmented, min_object_size)

        # reject noisy / low-quality segmentations that have somehow made it this far
        filled_volume_ratio = stack_segmented.sum() / stack_segmented.size
        if filled_volume_ratio > filled_volume_threshold:
            msg = (
                "Segmentation results are too noisy: segmented volume ratio "
                f"{filled_volume_ratio:.2f} above threshold "
                f"{filled_volume_threshold:.2f}."
            )
            raise ValueError(msg)

        self.is_segmented = True
        self.stack_segmented = stack_segmented > threshold
