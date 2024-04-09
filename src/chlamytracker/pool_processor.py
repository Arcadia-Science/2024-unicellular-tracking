import numpy as np
import skimage as ski

from .stack_processing import (
    circular_alpha_mask,
    gaussian_filter_3d_parallel,
    remove_small_objects_3d_parallel,
    rescale_to_float,
)
from .utils import timeit


class PoolSegmenter:
    """Class for processing timelapse microscopy data of an individual agar microchamber pool.

    TODO: detailed description of what processing steps this class seeks to accomplish.

    Parameters
    ----------
    raw_data_pool : (T, Y, X) uint16 array
        Input timelapse microscopy image data of an individual agar microchamber
        pool that has been tightly cropped to either manually or e.g. after
        being detected with `PoolFinder.find_pools()`.
    gaussian_filter_sigma : scalar (optional)
        Sigma of Gaussian filter for blurring the alpha mask (preprocessing).
    num_workers : int (optional)
        Number of processors to dedicate for multiprocessing.

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
        raw_data_pool,
        gaussian_filter_sigma=4,
        num_workers=6,
    ):
        self.raw_data = raw_data_pool.copy()
        self.gaussian_filter_sigma = gaussian_filter_sigma
        self.num_workers = num_workers

    def has_cells(self, contrast_threshold=0.05):
        """Determine whether pool contains cells.

        Determination is based on the amount of contrast in the standard
        deviation projection, using the variance of intensity values as a proxy
        for contrast.
        """
        # get dtype limits for normalization
        # (0, 65535) is expected but safer to check
        dtype_limit_max = max(ski.util.dtype_limits(self.raw_data))
        # compute the standard deviation projection
        std_intensity_projection = self.raw_data.std(axis=0)
        # use variance of intensity as measure of contrast
        normalized_contrast = std_intensity_projection.var() / dtype_limit_max
        return normalized_contrast > contrast_threshold

    @timeit
    def segment(self, min_area=150, filled_ratio_threshold=0.1):
        """"""
        # background subtraction
        background_subtracted = self.subtract_background()
        # segment cells based on Li thresholding -- more forgiving than Otsu
        threshold = ski.filters.threshold_li(background_subtracted)
        segmentation = background_subtracted > threshold

        # apply circular alpha mask to segmentation
        segmentation_masked = circular_alpha_mask(
            segmentation, num_workers=self.num_workers
        ).astype(bool)

        # filter out small objects
        segmentation_area_filtered = remove_small_objects_3d_parallel(
            segmentation_masked, min_area=min_area, num_workers=self.num_workers
        )

        # reject segmentations of noise
        filled_ratio = segmentation_area_filtered.sum() / segmentation_area_filtered.size
        if filled_ratio > filled_ratio_threshold:
            msg = (
                "Segmentation results are too noisy: segmented volume ratio "
                f"{filled_ratio:.2f} above threshold "
                f"{filled_ratio_threshold:.2f}."
            )
            raise ValueError(msg)

        return segmentation_area_filtered

    @timeit
    def subtract_background(self, sigma=1.6):
        """Apply background subtraction to the raw data.

        Parameters
        ----------
        sigma : float (optional)
            Standard deviation for Gaussian kernel.
        """
        mean_projection = self.raw_data.mean(axis=0)
        background_subtracted = np.clip(self.raw_data - mean_projection, -np.inf, 0)
        background_subtracted_rescaled = 1 - rescale_to_float(background_subtracted)
        background_subtracted_smoothed = gaussian_filter_3d_parallel(
            background_subtracted_rescaled, sigma=sigma, num_workers=self.num_workers
        )
        return background_subtracted_smoothed
