from multiprocessing import Pool as Workers
import numpy as np
import skimage as ski
import scipy.ndimage as ndi

from .utils import timeit


class Pool:
    """A 3D image stack representation of an agar microchamber pool.
    
    Parameters
    ----------
    stack : (Z, Y, X) array
    r_median : scalar
        Radius of median filter to apply (preprocessing).
    blur : scalar
        Sigma of Gaussian filter for blurring the alpha mask (preprocessing).

    Attributes
    ----------
    preprocessed : bool
    segmented : bool
    stack_raw : (Z, Y, X) array
        Raw (unprocessed) image stack.
    stack_prp : (Z, Y, X) array
        Pre-processed image stack (prior to segmentation).
    stack_seg : (Z, Y, X) array
        Segmented image stack.

    Methods
    -------
    preprocess()
    segment()
    """
    def __init__(
        self,
        stack,
        blur=4,
        r_median=4,
    ):
        # initialize Pool attributes
        self.stack_raw = stack
        self.blur = blur
        self.r_median = r_median

        self.preprocessed = False
        self.segmented = False

    @timeit
    def preprocess(
        self,
        remove_stationary_objects=True
    ):
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
          radius r = 1/2 width of prism.
        """

        # apply median filter
        pool_filtered = median_filter_3d_parellel(
            self.stack_raw,
            r_disk=self.r_median
        )

        # invert contrast
        pool_inverted = ski.util.invert(
            pool_filtered
        )

        # create alpha mask
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
            sigma=self.blur
        )
        # apply mask (prism --> tube)
        pool_tube = mask * pool_inverted

        # estimate background from a central column of intensity values
        z1, z2 = nz//2 - 50, nz//2 + 50
        y1, y2 = round(ny//2 - 0.3*ny), round(ny//2 + 0.3*ny)
        x1, x2 = round(nx//2 - 0.3*nx), round(nx//2 + 0.3*nx)
        column = pool_tube[z1:z2, y1:y2, x1:x2]
        bg = np.median(column)
        # subtract background by clipping mean intensity of central column
        pool_rescaled = ski.exposure.rescale_intensity(
            pool_tube,
            in_range=(bg, pool_tube.max()),
            out_range=(0, 1)
        )

        # remove junk but also non-motile cells by subtracting the mean
        # intensity projection
        if remove_stationary_objects:
            pool_rescaled -= pool_rescaled.mean(axis=0)
            pool_rescaled = np.clip(
                pool_rescaled, 0, pool_rescaled.max()
            )

        self.preprocessed = True
        self.stack_prp = pool_rescaled

    def segment(self):
        """Segment cells in preprocessed pool for tracking."""

        # if not self.preprocessed:
        #     self.preprocess(remove_stationary_objects=True)

        # self.segmented = True
        # self.stack_seg = ...

        msg = "Robust segmentation is still in the works..."
        raise NotImplementedError(msg)


def median_filter_3d_parellel(
    stack,
    r_disk=4,
    n_workers=6
):
    """Apply median filter to every image in a stack along the first axis 
    (in parallel).

    Notes
    -----
    * Timing analysis showed diminishing returns beyond 6 workers.
    """

    # make a bunch of footprints
    footprint = ski.morphology.disk(r_disk)
    footprints = [footprint]*stack.shape[0]
    # run median filter in parallel
    with Workers(n_workers) as ws:
        out = ws.starmap(ski.filters.median, zip(stack, footprints))

    return np.array(out)
