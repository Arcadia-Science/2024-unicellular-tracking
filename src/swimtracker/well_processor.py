from __future__ import annotations
import logging
import multiprocessing

import dask.array as da
import numpy as np
from dask_image import ndfilters
from numpy.typing import NDArray

from .stack_processing import (
    gaussian_filter_3d_parallel,
    otsu_threshold_3d,
    otsu_threshold_dask,
    remove_small_objects_3d_parallel,
    rescale_to_float,
)
from .timelapse import Timelapse
from .utils import timeit

logger = logging.getLogger(__name__)
NUM_WORKERS = multiprocessing.cpu_count() - 1


class WellSegmenter:
    """Segment timelapse microscopy data from a single well.

    Processes timelapse data from individual wells of multi-well plates (96 or 384 wells)
    using background subtraction, Gaussian filtering, and Otsu thresholding.

    Attributes:
        timelapse: The input timelapse data container.
        raw_data: Reference to the raw image data array.
        is_dask_array: Whether the data is stored as a dask array.
    """

    def __init__(self, timelapse: Timelapse) -> None:
        """Initialize the WellSegmenter.

        Args:
            timelapse: The timelapse data to process.
        """
        if timelapse is None:
            raise ValueError("Input timelapse cannot be None.")

        self.timelapse = timelapse
        self.raw_data = timelapse.raw_data
        self.is_dask_array = isinstance(self.raw_data, da.Array)

    @timeit
    def segment(
        self,
        sigma: float = 1.6,
        min_cell_diameter_um: float = 6,
        filled_ratio_threshold: float = 0.1,
        num_workers: int = NUM_WORKERS,
    ) -> NDArray[np.bool_]:
        """Segment cells in the timelapse data.

        Applies background subtraction, Gaussian smoothing, Otsu thresholding,
        and size filtering to identify cellular objects.

        Args:
            sigma: Standard deviation for Gaussian kernel.
            min_cell_diameter_um: Minimum cell diameter in micrometers for filtering.
            filled_ratio_threshold: Maximum allowed ratio of segmented pixels to total pixels.
            num_workers: Number of parallel workers to use.

        Returns:
            Binary segmentation mask with same spatial dimensions as input.

        Raises:
            ValueError: If segmentation results are too noisy (above filled_ratio_threshold).
        """
        if sigma <= 0:
            raise ValueError(f"Sigma must be positive, got {sigma}")
        if min_cell_diameter_um <= 0:
            raise ValueError(f"Minimum cell diameter must be positive, got {min_cell_diameter_um}")
        if not 0 < filled_ratio_threshold < 1:
            raise ValueError(
                f"Filled ratio threshold must be between 0 and 1, got {filled_ratio_threshold}"
            )
        if num_workers <= 0:
            raise ValueError(f"Number of workers must be positive, got {num_workers}")

        self.background_subtracted = self.subtract_background(sigma, num_workers)
        segmentation = self.apply_threshold()

        # Subsequent segmentation steps are not dask compatible --> convert dask array to numpy
        if self.is_dask_array:
            segmentation = segmentation.compute()

        # Filter out small objects
        min_area = self.timelapse.convert_um_to_px2_circle(min_cell_diameter_um)
        segmentation_area_filtered = remove_small_objects_3d_parallel(
            segmentation, min_area=min_area, num_workers=num_workers
        )

        # Reject segmentations of noise
        filled_ratio = segmentation_area_filtered.sum() / segmentation_area_filtered.size
        if filled_ratio > filled_ratio_threshold:
            raise ValueError(
                "Segmentation results are too noisy: segmented volume ratio "
                f"{filled_ratio:.2f} above threshold "
                f"{filled_ratio_threshold:.2f}."
            )

        return segmentation_area_filtered

    @timeit
    def subtract_background(
        self, sigma: float, num_workers: int
    ) -> NDArray[np.floating] | da.Array:
        """Apply background subtraction to the raw data.

        Computes mean projection across time, subtracts it from each frame,
        rescales to [0,1], and applies Gaussian smoothing.

        Args:
            sigma: Standard deviation for Gaussian kernel.
            num_workers: Number of parallel workers (ignored for dask arrays).

        Returns:
            Background-subtracted and smoothed image data.

        Note:
            Smoothing uses dask for dask arrays and scikit-image for numpy arrays.
        """
        mean_projection = self.raw_data.mean(axis=0)
        background_subtracted = np.clip(self.raw_data - mean_projection, -np.inf, 0)
        background_subtracted_rescaled = 1 - rescale_to_float(background_subtracted)

        if self.is_dask_array:
            background_subtracted_smoothed = ndfilters.gaussian_filter(
                background_subtracted_rescaled, sigma=sigma
            )
        else:
            background_subtracted_smoothed = gaussian_filter_3d_parallel(
                background_subtracted_rescaled, sigma=sigma, num_workers=num_workers
            )

        return background_subtracted_smoothed

    @timeit
    def apply_threshold(self) -> NDArray[np.bool_] | da.Array:
        """Apply Otsu thresholding to segment the data.

        Uses the background-subtracted data to compute an optimal threshold
        and create a binary segmentation mask.

        Returns:
            Binary segmentation mask where True indicates foreground pixels.

        Raises:
            RuntimeError: If background subtraction hasn't been performed yet.

        Note:
            Uses dask-compatible thresholding for dask arrays.
        """
        if not hasattr(self, "background_subtracted"):
            raise RuntimeError("Must call subtract_background() before apply_threshold()")

        # Otsu thresholding
        if self.is_dask_array:
            threshold = otsu_threshold_dask(self.background_subtracted)
        else:
            threshold = otsu_threshold_3d(self.background_subtracted)

        segmentation = self.background_subtracted > threshold
        return segmentation
