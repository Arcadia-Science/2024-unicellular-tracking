import numpy as np
from dask_image import ndfilters

from .stack_processing import (
    gaussian_filter_3d_parallel,
    otsu_threshold_3d,
    otsu_threshold_dask,
    remove_small_objects_3d_parallel,
    rescale_to_float,
)
from .timelapse import Timelapse
from .utils import timeit


class WellSegmenter(Timelapse):
    """Subclass of `Timelapse` for segmenting timelapse microscopy data from one
    well of a 384 or 1536 well plate."""

    def __init__(self, nd2_file, use_dask=False):
        super().__init__(nd2_file, use_dask)

    @timeit
    def segment(self, min_cell_diameter_um=6, filled_ratio_threshold=0.1):
        """"""
        # background subtraction
        background_subtracted = self.subtract_background()

        # different segmentation approaches depending on whether or not a zstack
        if self.is_zstack:
            segmentation = self._segment_zstack(background_subtracted)
        else:
            segmentation = self._segment_zslice(background_subtracted)

        # subsequent segmentation steps are not dask compatible
        if self.use_dask:
            segmentation = segmentation.compute()

        # filter out small objects
        min_area = self.convert_um_to_px2_circle(min_cell_diameter_um)
        segmentation_area_filtered = remove_small_objects_3d_parallel(
            segmentation, min_area=min_area, num_workers=10
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

        Smoothing is effectively and mathematically the same whether using
        the skimage or dask api, only change is in how the computation is done
        (number of cores used, when it is executed, etc.).

        Parameters
        ----------
        sigma : float (optional)
            Standard deviation for Gaussian kernel.
        """
        mean_projection = self.raw_data.mean(axis=0)
        background_subtracted = np.clip(self.raw_data - mean_projection, -np.inf, 0)
        background_subtracted_rescaled = 1 - rescale_to_float(background_subtracted)

        if self.use_dask:
            background_subtracted_smoothed = ndfilters.gaussian_filter(
                background_subtracted_rescaled, sigma=sigma
            )
        else:
            background_subtracted_smoothed = gaussian_filter_3d_parallel(
                background_subtracted_rescaled, sigma=sigma, num_workers=10
            )

        return background_subtracted_smoothed

    @timeit
    def _segment_zslice(self, background_subtracted):
        """"""
        # otsu threholding
        if self.use_dask:
            threshold = otsu_threshold_dask(background_subtracted)
        else:
            threshold = otsu_threshold_3d(background_subtracted)

        segmentation = background_subtracted > threshold
        return segmentation

    def _segment_zstack(self, background_subtracted):
        """"""
        raise NotImplementedError("Cannot yet segment a zstack timelapse.")
        # std_projection = self.raw_data.std(axis=1)
        # thresh = np.percentile(std_projection, 99)
        # segmentation = std_projection > thresh
        # return segmentation
