from functools import partial

import nd2
import numpy as np
import skimage as ski

from .pool_processor import MicrochamberPoolProcessor
from .utils import timeit


class PoolFinder:
    """Class for processing timelapse brightfield microscopy data of agar microchamber pools.

    TODO: more detailed description of what an agar microchamber pool is and what
          processing steps this class seeks to accomplish (and why).

    Parameters
    ----------
    filepath : `pathlib.Path`
        Path to input nd2 file.
    pool_radius_um : scalar
        Radius of pool in microns.
    pool_spacing_um : scalar
        (Center <--> center) distance between pools in microns.
    hough_threshold : float (optional)
        Threshold for Hough transformation (0, 1)
    median_filter_radius : scalar (optional)
        Radius of structuring element for median filter during preprocessing.
    tophat_filter_radius : scalar (optional)
        Radius of structuring element for tophat filter during preprocessing.
    min_object_size : scalar (optional)
        Area threshold for object removal during preprocessing (objects below
        this size will be removed).

    Attributes
    ----------
    stack : (T, Y, X) array
        Timelapse data as numpy array.
    pool_radius_px : float
        Radius of pool in pixels.
    pool_spacing_px : float
        (Center <--> center) distance between pools in pixels.
    max_num_pools : int
        Max number of pools in timelapse as allowed by geometry.
    mean_intensity_projection : (Y, X) array
        Mean intensity projection of timelapse.
    pool_edges : (Y, X) array
        Mean intensity projection of timelapse thresholded to enhance edges
        for circle detection.
    poolmap : dict
        Mapping of grid points to x, y coordinates of pool locations.
    pools : dict
        Collection of `MicrochamberPoolProcessor`s.
    """

    def __init__(
        self,
        filepath,
        pool_radius_um,
        pool_spacing_um,
        hough_threshold=0.2,
        median_filter_radius=2,
        tophat_filter_radius=5,
        min_object_size=500,
    ):
        # check that nd2 file is valid
        self.filepath = filepath
        self._validate()

        # metadata from .nd2 headers
        with nd2.ND2File(filepath) as nd2f:
            voxels_um = nd2f.voxel_size()  # in microns
            sizes = nd2f.sizes  # e.g. {'T': 10, 'C': 2, 'Y': 256, 'X': 256}
        self.Nx = sizes["X"]
        self.Ny = sizes["Y"]
        self.um_per_px = (voxels_um.x + voxels_um.y) / 2

        # pre-processing parameters
        self.median_filter_radius = median_filter_radius
        self.tophat_filter_radius = tophat_filter_radius
        self.min_object_size = min_object_size
        self.is_preprocessed = False

        # Hough transform parameters
        self.pool_radius_px = pool_radius_um / self.um_per_px
        self.pool_spacing_px = pool_spacing_um / self.um_per_px
        self.max_num_pools = self.get_max_num_pools()
        self.hough_threshold = hough_threshold

        # initialize empty mapping of grid indices to coordinates of pool locations
        #   grid is quite generous in that an extra row and column is generated
        #   beyond what should be allowed by the image dimensions; this allows
        #   for flexibility in detecting pools which is later accounted for by
        #   raising an IndexError for pools that are found to be out of bounds.
        nx = int(np.ceil(self.Nx / self.pool_spacing_px))
        ny = int(np.ceil(self.Ny / self.pool_spacing_px))
        self.grid = np.mgrid[-1 : nx + 1, -1 : ny + 1].reshape(2, -1).T
        self.poolmap = dict.fromkeys([(ix, iy) for (ix, iy) in self.grid])

        # load data from nd2 file
        self.load()

    def _validate(self):
        """Check that nd2 file has not been corrupted."""
        try:
            # simply checking for shape will determine if nd2 file is corrupted
            with nd2.ND2File(self.filepath) as nd2f:
                _ = nd2f.shape
        except ValueError as err:
            msg = f"{self.filepath} is corrupted."
            print(msg, err)

    @timeit
    def load(self):
        """Load timelapse from nd2 file as numpy array."""
        if not hasattr(self, "stack"):
            self.stack = nd2.imread(self.filepath)

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
        Nx = self.Nx
        Ny = self.Ny
        d = self.pool_spacing_px

        # number of pools in each dimension allowing for partial pools
        n_pools_x = int(np.ceil(Nx / d))
        n_pools_y = int(np.ceil(Ny / d))

        return n_pools_x * n_pools_y

    @timeit
    def preprocess(self):
        """Apply preprocessing steps

        Process
        -------
        1) Calculate the mean intensity projection (along time axis)
        2) Median filter
        3) Clip intensity range
        4) White tophat filter
        5) Threshold
        6) Clean / remove small segments

        Returns
        -------
        mean_intensity_projection : (Y, X) array
            Mean intensity projection of timelapse.
        pool_edges : (Y, X) array
            Processed mean intensity projection with edges of pools enhanced (ideally).
        """
        # create structuring elements (aka footprint) for filters
        footprint_median_filter = ski.morphology.disk(self.median_filter_radius)
        footprint_tophat_filter = ski.morphology.disk(self.tophat_filter_radius)

        # calculate the mean intensity projection along the time axis
        tproj = self.stack.mean(axis=0)

        # apply edge-preserving smoothing filter
        tproj_smooth = ski.filters.median(tproj, footprint=footprint_median_filter)

        # clip intensity range (auto- brightness/contrast)
        # NOTE: normally would set intensity range for auto- brightness/contrast
        #       based on e.g. (1%, 99%) percentile range, but artefacts in the
        #       microscopy data caused too many inconsistencies with this
        #       approach so opted to clip intensity based on median +/- k*std
        #       where k is somewhat arbitrary but there are no objective
        #       answers here...
        med = np.median(tproj_smooth)
        std = tproj_smooth.std()
        vmin, vmax = (med - 2 * std, med + 2 * std)
        tproj_rescaled = ski.exposure.rescale_intensity(tproj_smooth, in_range=(vmin, vmax))
        # tophat filter
        tproj_pool_edges = ski.morphology.white_tophat(
            image=tproj_rescaled, footprint=footprint_tophat_filter
        )
        # create rough mask on edges of the pools
        thresh = ski.filters.threshold_otsu(tproj_pool_edges)
        pool_edges = ski.morphology.remove_small_objects(
            tproj_pool_edges > thresh, min_size=self.min_object_size
        )

        self.is_preprocessed = True
        self.mean_intensity_projection = tproj_rescaled
        self.pool_edges = pool_edges

    def detect_pools(self):
        """Run Hough transform on preprocessed timelapse to detect pools.

        This is a first pass at detecting pools using a Hough transform [1]
        to detect circles of a given range of radii.

        References
        ----------
        [1] https://scikit-image.org/docs/stable/auto_examples/edges/plot_circular_elliptical_hough_transform.html
        """
        # preprocess
        if not self.is_preprocessed:
            self.preprocess()

        # define radii for Hough transform
        #   a 10px range around the expected radius was found to work empirically
        #   defining the range of radii in this way ensures a +/-5 window around
        #   the expected radius
        r_hough = range(round(self.pool_radius_px) - 5, round(self.pool_radius_px) + 6, 2)

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
        self.model = model

    @timeit
    def find_pools(self, residual_threshold=10):
        """Detect and extrapolate pool locations to populate `poolmap`.

        Runs detect_pools() and extrapolate_pool_locations(). Then updates
        `poolmap` by overwriting extrapolated pool locations with inliers.
        """

        # detect and extrapolate pool locations
        centers = self.detect_pools()
        self.extrapolate_pool_locations(centers)

        # map grid indices to extrapolated coordinates with `extrapolated` status
        for ix, iy in self.grid:
            cx, cy = self.model((ix, iy)).ravel().round().astype(int)
            self.poolmap[(ix, iy)] = [(cx, cy), "extrapolated"]

        # loop through detected pool locations to update the poolmap
        detected_grid_coords = self.model.inverse(centers).round()
        for i, (ix, iy) in enumerate(detected_grid_coords):
            # measure the distance between the detected location and the
            # estimated location for outlier detection
            # NOTE: safe to assume that (ix, iy) exists in self.poolmap since
            #       1) the grid is a dense grid of all integers and
            #       2) the bounds used to construct the grid mean that it's
            #          safe to assume that none of the inverse centers lie
            #          outside of the grid.
            cx, cy = self.poolmap[(ix, iy)][0]
            measured_residuals = self.model.residuals(
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
        """Extract pools."""
        # find pools
        self.find_pools()

        # extract pools
        pools = {}
        for (ix, iy), ((cx, cy), _status) in self.poolmap.items():
            # crop to pool (+1 pixel margin)
            try:
                pool_stack = crop_out_roi(
                    stack=self.stack, center=(cx, cy), radius=self.pool_radius_px + 1
                )
            # pool extends beyond image border --> skip
            except IndexError:
                continue

            # collect pools
            pool = MicrochamberPoolProcessor(pool_stack)
            pools[(ix, iy)] = pool

        self.pools = pools

    def preprocess_pools(self):
        """Apply MicrochamberPoolProcessor.prepocess() to each pool."""
        # run preprocessing on each pool and update collection
        # TODO: run in parallel (if possible)
        for (ix, iy), pool in self.pools.items():
            pool.preprocess()
            self.pools[(ix, iy)] = pool

    def segment_pools(self):
        """Apply MicrochamberPoolProcessor.segment() to each pool."""
        # segment each pool and update collection
        for (ix, iy), pool in self.pools.items():
            pool.segment(min_object_size=self.min_object_size)
            self.pools[(ix, iy)] = pool

    @timeit
    def export_pools(self, dir_out=None):
        """Export processed pools to disk as 8bit tiff stacks."""

        # set default output directory
        if dir_out is None:
            dir_out = self.filepath.parent / "processed"

        # loop through pools and save as uint8 tiffs
        # TODO: how to select which stack to export without hardcoding?
        for (ix, iy), pool in self.pools.items():
            # only export pools with cells
            if pool.has_cells():
                # convert to 8bit
                pool_8bit = ski.exposure.rescale_intensity(
                    pool.stack_segmented, in_range=(0, 1), out_range=(0, 255)
                ).astype(np.ubyte)

                # include pool x, y indices in filename
                tgt = dir_out / self.filepath.stem / f"pool_{ix:02d}_{iy:02d}.tiff"
                tgt.parent.mkdir(exist_ok=True, parents=True)
                ski.io.imsave(tgt, pool_8bit)

    def make_debug_sketch(self, save=True, dir_out=None):
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

        # save debug image to disk
        if save:
            if dir_out is None:
                dir_out = self.filepath.parent / "processed"
            # save as jpeg
            tgt = dir_out / (self.filepath.stem + "_pools.jpg")
            tgt.parent.mkdir(exist_ok=True, parents=True)
            ski.io.imsave(tgt, sketch)

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


def crop_out_roi(stack, center, radius):
    """Crops a square ROI out of an image stack along the first axis.

    Parameters
    ----------
    stack : ([T, Z], Y, X) array
        Image stack such as a timelapse or z-stack.
    center : 2-tuple
        ROI center as an (x, y) coordinate.
    radius : scalar
        Radius to determine cropping window (1/2 width of square).

    Returns
    -------
    roi : (Z, Y, X) array
        Region of interest cropped from image stack with dimensions (Z, 2*R, 2*R).

    Raises
    ------
    IndexError
        If requested crop is outside the extent of the stack.
    """
    # validate input
    cx, cy = tuple(int(i) for i in center)
    r = round(radius)

    # crop to a rectangular roi
    nz, ny, nx = stack.shape
    y1, y2 = cy - r, cy + r
    x1, x2 = cx - r, cx + r
    if (y1 < 0) or (y2 > ny) or (x1 < 0) or (x2 > nx):
        msg = (
            f"Requested crop (array[:, {y1}:{y2}, {x1}:{x2}]) is out of bounds "
            f"for array with shape {stack.shape}."
        )
        raise IndexError(msg)
    else:
        roi = stack[:, y1:y2, x1:x2]

    return roi
