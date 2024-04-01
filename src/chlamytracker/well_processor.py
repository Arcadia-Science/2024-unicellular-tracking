import logging

import numpy as np
import skimage as ski

from .timelapse import Timelapse
from .utils import timeit

logger = logging.getLogger(__name__)


class WellProcessor(Timelapse):
    """Subclass for processing timelapse microscopy data from one well of a 384
    or 1536 well plate."""

    def __init__(self, nd2_file):
        super().__init__(nd2_file)

        self.is_zstack = self.dimensions.get("Z") is not None

    @timeit
    def segment(self, min_cell_area=64):
        """"""
        # background subtraction
        logger.info("Begin background subtraction...")
        background_subtracted = self.subtract_background()
        logger.info("Background subtraction complete.")

        # different segmentation approaches depending on whether or not a zstack
        if self.is_zstack:
            logger.info("Begin zstack timelapse segmentation...")
            segmentation = self._segment_zstack(background_subtracted)
        else:
            logger.info("Begin timelapse segmentation...")
            segmentation = self._segment_zslice(background_subtracted)
        logger.info("Segmentation complete.")

        # filter out small objects
        logger.info(f"Removing objects smaller than {min_cell_area} px^2...")
        segmentation_area_filterd = np.array(
            [ski.morphology.remove_small_objects(im, min_size=min_cell_area) for im in segmentation]
        )
        logger.info("Small object removal complete.")

        return segmentation_area_filterd

    def subtract_background(self):
        """"""
        mean_projection = self.raw_data.mean(axis=0)
        background_subtracted = np.clip(self.raw_data - mean_projection, -np.inf, 0)
        background_subtracted_rescaled = ski.exposure.rescale_intensity(
            background_subtracted, out_range=(0, 1)
        )
        background_subtracted_inverted = 1 - background_subtracted_rescaled
        background_subtracted_smoothed = ski.filters.gaussian(
            background_subtracted_inverted, sigma=1.6
        )
        return background_subtracted_smoothed

    def _segment_zslice(self, background_subtracted):
        """"""
        thresh = ski.filters.threshold_otsu(background_subtracted)
        segmentation = background_subtracted > thresh
        return segmentation

    def _segment_zstack(self, background_subtracted):
        """"""
        raise NotImplementedError("Cannot yet segment a zstack timelapse.")
        # std_projection = self.raw_data.std(axis=1)
        # thresh = np.percentile(std_projection, 99)
        # segmentation = std_projection > thresh
        # return segmentation
