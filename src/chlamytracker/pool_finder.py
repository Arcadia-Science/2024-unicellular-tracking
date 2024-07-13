import logging
from functools import partial

import numpy as np
import skimage as ski

from .pool_processor import PoolSegmenter
from .stack_processing import crop_out_roi
from .timelapse import Timelapse
from .utils import timeit

logger = logging.getLogger(__name__)


class PoolFinder(Timelapse):
    """Subclass of `Timelapse` for detecting agar microchamber pools in
    timelapse microscopy data.

    TODO: more detailed description of what an agar microchamber pool is and what
          processing steps this class seeks to accomplish (and why).

    Parameters
    ----------
    nd2_file : Path | str
        Filepath to input nd2 file.
    pool_radius_um : int (optional)
        Radius of pool in microns.
    pool_spacing_um : int (optional)
        (Center <--> center) distance between pools in microns.
    min_cell_diameter_um : int (optional)
        Diameter of smallest desired organism to be segmented in microns.
    hough_threshold : float (optional)
        Threshold for Hough transformation (0, 1)
    min_object_size : int (optional)
        Area threshold for object removal after Sobel edge filtering and before
        applying the Hough transform.
    use_dask : bool (optional)
        Whether to load and process nd2 file with dask.
    """

    def __init__(
        self,
        nd2_file,
        pool_radius_um=50,
        pool_spacing_um=200,
        min_cell_diameter_um=6,
        hough_threshold=0.2,
        min_object_size=400,
        use_dask=False,
        load=True,
    ):
        super().__init__(nd2_file, use_dask, load)

        # object removal parameters
        self.min_cell_diameter_um = min_cell_diameter_um
        self.min_object_size = min_object_size

        # Hough transform parameters
        self.pool_radius_px = pool_radius_um / self.pixelsize_um
        self.pool_spacing_px = pool_spacing_um / self.pixelsize_um
        self.max_num_pools = self.get_max_num_pools()
        self.hough_threshold = hough_threshold

        # initialize empty mapping of grid indices to coordinates of pool locations
        #   grid is quite generous in that an extra row and column is generated
        #   beyond what should be allowed by the image dimensions; this allows
        #   for flexibility in detecting pools which is later accounted for by
        #   raising an IndexError for pools that are found to be out of bounds.
        nx = int(np.ceil(self.dimensions["X"] / self.pool_spacing_px))
        ny = int(np.ceil(self.dimensions["Y"] / self.pool_spacing_px))
        self.grid = np.mgrid[-1 : nx + 1, -1 : ny + 1].reshape(2, -1).T
        self.poolmap = dict.fromkeys([(ix, iy) for (ix, iy) in self.grid])

    def get_max_num_pools(self):
        """Calculate the max number of pools in the timelapse based on geometry.

        The real maximum number of pools in either dimension is

            n_pools_x = int(Nx >= 2*r) + (Nx - 2*r) // d
            n_pools_y = int(Ny >= 2*r) + (Ny - 2*r) // d

        (where r = pool_radius_px) as this is the number of whole pools that
        can fit within the dimensions of the image. But because the Hough
        transform can detect partial pools (rings), the calculation here returns
        the more practical maximum for subsequent circle detection.
        """
        Nx = self.dimensions["X"]
        Ny = self.dimensions["Y"]
        d = self.pool_spacing_px

        # number of pools in each dimension allowing for partial pools
        n_pools_x = int(np.ceil(Nx / d))
        n_pools_y = int(np.ceil(Ny / d))

        return n_pools_x * n_pools_y

    @timeit
    def get_pool_edges(self):
        """Run edge detection on timelapse data to get outlines of the pools.

        Uses Sobel edge detection on the mean intensity projection of the
        timelapse to find edges of the agar microchamber pools. Edges are
        cleaned up by filtering out small objects (junk) with
        `ski.remove.remove_small_objects`.
        """
        # mean intensity projection
        mean_projection = self.raw_data.mean(axis=0)

        # enhance contrast -- clip intensity centered on the median
        med = np.median(mean_projection)
        std = mean_projection.std()
        vmin, vmax = med - 2 * std, med + 2 * std
        mean_projection_rescaled = ski.exposure.rescale_intensity(
            mean_projection, in_range=(vmin, vmax)
        )

        # edge detection on mean intensity projection
        edges = ski.filters.sobel(mean_projection_rescaled)
        threshold = ski.filters.threshold_otsu(edges)
        edges_binary = edges > threshold

        # remove junk
        pool_edges = ski.morphology.remove_small_objects(
            edges_binary, min_size=self.min_object_size
        )
        self.pool_edges = pool_edges

    def detect_pools(self):
        """Run Hough transform on preprocessed timelapse to detect pools.

        This is a first pass at detecting pools using a Hough transform [1]
        to detect circles of a given range of radii.

        References
        ----------
        [1] https://scikit-image.org/docs/stable/auto_examples/edges/plot_circular_elliptical_hough_transform.html
        """
        # get rough outlines of each pool
        self.get_pool_edges()

        # define radii for Hough transform
        #   a 10px range around the expected radius was found to work empirically
        #   defining the range of radii in this way ensures a +/-5 window around
        #   the expected radius
        r_hough = range(
            round(self.pool_radius_px) - 5,
            round(self.pool_radius_px) + 6,
            2,
        )

        # set minimum search distance between adjacent circles
        d_min = int(0.9 * self.pool_spacing_px)

        # apply circular Hough transform
        hspaces = ski.transform.hough_circle(self.pool_edges, r_hough)
        hough_peaks = ski.transform.hough_circle_peaks(
            hspaces=hspaces,
            radii=r_hough,
            min_xdistance=d_min,
            min_ydistance=d_min,
            total_num_peaks=self.max_num_pools,
            threshold=self.hough_threshold,
        )

        # unpack Hough peaks
        centers = np.stack(
            [
                hough_peaks[1],  # x coordinates
                hough_peaks[2],  # y coordinates
            ]
        ).T

        return centers

    def extrapolate_pool_locations(self, centers):
        """Extrapolate pool locations --> "extrapoolate".

        The idea here is that some (but not all) pools have been found by
        running `detect_pools()`. Now we want to find the remaining pools
        which could have gone undetected for any number of reasons but
        essentially all boil down to imperfect microscopy data: blurred edges
        due to focus plane issues, artefacts or junk obscuring the pools,
        or just weak contrast (some of these issues are reasons to discard
        pools for further processing, but some are not... kind of tricky).

        Regardless of the merits, we can estimate the center coordinates of
        each pool in the timelapse based on the coordinates of those that have
        already been detected via a least squares fit and the a priori
        knowledge that the pools are arranged in a grid. We seek the linear
        transformation that transforms the coordinates of a simple grid layout
        e.g.

            (0, 0)  (1, 0)  (2, 0)
            (0, 1)  (1, 1)  (2, 1)
            (0, 2)  (1, 2)  (2, 2)

        to the center coordinates of each pool e.g.

            (56, 197)  (358, 170)  (662, 146)
            (74, 500)  (379, 477)  (684, 451)
            (94, 806)  (401, 781)  (706, 756)

        This linear transformation can be solved for using the RANSAC (random
        sample consensus) algorithm [1], which is robust against outliers,
        which is beneficial here because `detect_pools()` will occasionally
        return false positives.

        References
        ----------
        [1] https://en.wikipedia.org/wiki/Random_sample_consensus
        """
        # minimum number of points required to fit a Similarity tranformation
        min_samples = 3
        if len(centers) < min_samples:
            err = (
                f"Unable to extrapolate pool locations from only "
                f"{len(centers)} detected pools ({min_samples} needed)."
            )
            raise ValueError(err)

        # bin the coordinates of the detected pools into cells of a grid
        # NOTE: this is unfortunately not very accurate as the grid
        #       of pools is often rotated in the image, but it seems to work
        #       well enough empirically at least as a starting point
        src = centers // self.pool_spacing_px

        # the Similarity transformation for inferring the coordinates of
        # the non-detected pools should have a scale approximately equal
        # to the distance (in pixels) between each pool. Therefore reject
        # all models with scales << or >> than this expected scale.
        pct_diff = 0.05
        min_scale = (1 - pct_diff) * self.pool_spacing_px
        max_scale = (1 + pct_diff) * self.pool_spacing_px
        is_model_valid = partial(validate_model, min_scale=min_scale, max_scale=max_scale)

        # use RANSAC to find the linear transformation (w/o shear) that
        # maps the grid indices to the coordinates of the detected pools
        model, _inliers = ski.measure.ransac(
            data=(src, centers),
            model_class=ski.transform.SimilarityTransform,
            min_samples=min_samples,
            residual_threshold=10,
            max_trials=500,
            is_model_valid=is_model_valid,
        )
        return model

    @timeit
    def find_pools(self, residual_threshold=10):
        """Detect and extrapolate pool locations to populate `poolmap`.

        Runs detect_pools() and extrapolate_pool_locations(). Then updates
        `poolmap` by overwriting extrapolated pool locations with inliers.
        """
        # detect and extrapolate pool locations
        centers = self.detect_pools()
        model = self.extrapolate_pool_locations(centers)

        # map grid indices to extrapolated coordinates with `extrapolated` status
        for ix, iy in self.grid:
            cx, cy = model((ix, iy)).ravel().round().astype(int)
            self.poolmap[(ix, iy)] = [(cx, cy), "extrapolated"]

        # loop through detected pool locations to update the poolmap
        detected_grid_coords = model.inverse(centers).round()
        for i, (ix, iy) in enumerate(detected_grid_coords):
            # measure the distance between the detected location and the
            # estimated location for outlier detection
            # NOTE: safe to assume that (ix, iy) exists in self.poolmap since
            #       1) the grid is a dense grid of all integers and
            #       2) the bounds used to construct the grid mean that it's
            #          safe to assume that none of the inverse centers lie
            #          outside of the grid.
            cx, cy = self.poolmap[(ix, iy)][0]
            measured_residuals = model.residuals(
                (ix, iy),  # detected
                (cx, cy),  # extrapolated
            ).item()

            # overwrite extrapolated locations with detected locations if
            # residuals are below a given threshold
            if measured_residuals < residual_threshold:
                status = "inlier"
                self.poolmap[(ix, iy)] = [tuple(centers[i, :]), status]

    @timeit
    def extract_pools(self):
        """Extract pools.

        Returns a mapping of grid coordinates of pool locations to `PoolSegmenter` instances.
        >>> self.extract_pools()
            {
                (0, 0): `PoolSegmenter`,
                (0, 1): `PoolSegmenter`,
                (0, 2): `PoolSegmenter`,
                ...
                (Nx, Ny): `PoolSegmenter`
            }
        """
        # find pools --> updates self.poolmap
        self.find_pools()

        # extract pools
        pools = {}
        for (ix, iy), ((cx, cy), _status) in self.poolmap.items():
            # crop to pool (+1 pixel margin)
            try:
                raw_data_pool = crop_out_roi(
                    stack=self.raw_data, center=(cx, cy), radius=self.pool_radius_px + 1
                )
            # pool extends beyond image border --> skip
            except IndexError:
                continue

            # collect pools
            pools[(ix, iy)] = PoolSegmenter(raw_data_pool)

        return pools

    def segment_pools(self, filled_ratio_threshold=0.1):
        """Segment cells from each of the detected pools.

        Returns a mapping of grid coordinates of pool locations to segmentation data.
        >>> self.segment_pools(*args)
            {
                (0, 0): (T, Y, X) bool array,
                (0, 1): (T, Y, X) bool array,
                (0, 2): (T, Y, X) bool array,
                ...
                (Nx, Ny): (T, Y, X) bool array
            }
        """
        # set imaging modality for segmentation
        modality = "brightfield" if self.is_brightfield else "DIC"

        # convert minimum cell diameter to pixelated area
        min_area = self.convert_um_to_px2_circle(self.min_cell_diameter_um)

        pools = self.extract_pools()
        pools_segmented = {}
        # segment each pool and update collection
        for (ix, iy), pool in pools.items():
            # only bother segmenting if the pool contains cells
            if pool.has_cells():
                try:
                    pool_segmented = pool.segment(
                        modality=modality,
                        min_area=min_area,
                        filled_ratio_threshold=filled_ratio_threshold,
                    )

                except ValueError as err:
                    msg = f"Processing for pool ({ix}, {iy}) in {self.nd2_file.name} failed: "
                    logger.error(msg + str(err))
                    continue

                pools_segmented[(ix, iy)] = pool_segmented

        return pools_segmented

    def make_debug_sketch(self):
        """Annotates the detected pools for debugging purposes."""

        colormap = {
            "inlier": (20, 255, 255),  # cyan
            "outlier": (255, 150, 20),  # orange
            "extrapolated": (20, 255, 20),  # green
        }

        # convert to 8bit rgb image
        sketch = ski.color.gray2rgb((0.7 * 255 * self.pool_edges).astype(np.ubyte))
        for (_ix, _iy), ((cx, cy), status) in self.poolmap.items():
            # annotate circle outline
            rr, cc = ski.draw.circle_perimeter(
                r=cy, c=cx, radius=int(self.pool_radius_px), shape=sketch.shape
            )
            sketch[rr, cc] = colormap[status]

            # annotate circle center
            rr, cc = ski.draw.disk(center=(cy, cx), radius=6, shape=sketch.shape)
            sketch[rr, cc] = colormap[status]

        return sketch


def validate_model(model, src, dst, min_scale, max_scale):
    """Validate RANSAC model with upper and lower bounds for the scale.

    This function gets passed to `skimage.measure.ransac` for use in
    determining whether or not an estimated model is valid or not.

    References
    ----------
    [1] https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.ransac
    """
    is_valid = min_scale < model.scale < max_scale
    return is_valid
