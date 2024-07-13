import numpy as np

from .stack_processing import (
    circular_alpha_mask,
    gaussian_filter_3d_parallel,
    get_central_frames,
    li_threshold_3d,
    otsu_threshold_3d,
    remove_small_objects_3d_parallel,
    rescale_to_float,
)
from .utils import timeit


class PoolSegmenter:
    """Class for processing timelapse microscopy data of an individual agar microchamber pool.

    Primary application is for segmenting cells within an individual agar
    microchamber pool [1]. Performs background subtraction prior to segmentation.
    Background is estimated as the mean intensity projection.

    Parameters
    ----------
    raw_data_pool : (T, Y, X) uint16 array
        Input timelapse microscopy image data of an individual agar microchamber
        pool that has been tightly cropped to either manually or e.g. after
        being detected with `PoolFinder.find_pools()`.
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

    def __init__(self, raw_data_pool, num_workers=6):
        self.raw_data = raw_data_pool.copy()
        self.num_workers = num_workers

    def has_cells(self, contrast_threshold=1e-6):
        """Determine whether pool contains cells.

        Determination is based on the amount of contrast in the standard
        deviation projection, using the variance of intensity values as a proxy
        for contrast.
        """
        # get dtype limits for normalization
        # (0, 65535) is expected but safer to check
        # dtype_limit_max = max(ski.util.dtype_limits(self.raw_data))
        dtype_limit_max = 1
        # std projection on smoothed substack
        num_frames = min(200, self.raw_data.shape[0])
        central_frames = get_central_frames(self.raw_data, num_frames)
        central_frames_smoothed = gaussian_filter_3d_parallel(central_frames, sigma=3).std(axis=0)
        std_projection = central_frames_smoothed.std(axis=0)
        # use variance of intensity as measure of contrast
        normalized_contrast = std_projection.var() / dtype_limit_max
        return normalized_contrast > contrast_threshold

    @timeit
    def segment(
        self,
        modality="brightfield",
        min_area=150,
        filled_ratio_threshold=0.1,
        li_threshold=0.1,
        otsu_thresholding_scale_factor=0.66,
    ):
        """Segment cells within a pool.

        Parameters
        ----------
        modality : str (optional)
            Imaging modality: either "brightfield" or "DIC".
        min_area : float (optional)
            Minimum area (px^2) for object removal.
        filled_ratio_threshold : float (optional)
            Threshold used for discarding noisy segmentation results.
            For reliable cell tracking it is assumed that the pools are
            quite sparsely populated with cells.
        li_threshold : float (optional)
            Initial guess parameter for Li thresholding [1].
        otsu_thresholding_scale_factor : float (optional)
            Value for (somewhat arbitrarily) scaling the Otsu threshold.
            Default value of 0.66 was found to work empirically on a sample
            of test DIC time lapses.

        References
        ----------
        [1] https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_li
        """
        # background subtraction
        background_subtracted = self.subtract_background()

        # set threshold for segmentation based on modality -- Li thresholding
        # was empirically determined to provide better results on brightfield
        # microscopy data, while Otsu performed better on DIC
        if modality == "brightfield":
            threshold = li_threshold_3d(background_subtracted, initial_guess=li_threshold)
        else:
            threshold = otsu_threshold_3d(background_subtracted) * otsu_thresholding_scale_factor

        # apply threshold
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
