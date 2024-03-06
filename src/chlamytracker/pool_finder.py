import nd2
import numpy as np
import skimage as ski

from .pool_processor import MicrochamberPoolProcessor
from .utils import timeit


class PoolFinder:
    """Class for detecting and extracting pools from timelapse image data.

    Parameters
    ----------
    filepath : `pathlib.Path`
        Path for input nd2 file.
    r_pool_um : scalar (optional)
        Radius of pool in microns.
    d_bw_pools_um : scalar (optional)
        (Center <--> center) distance between pools in microns.
    t_hough : float (optional)
        Threshold for Hough transformation (0, 1)
    projection : str (optional)
        Type of intensity projection to apply over timelapse.
    r_median : scalar (optional)
        Radius of footprint for median filtering.
    r_tophat : scalar (optional)
        Radius of footprint for tophat filtering.
    s_rm_objs : scalar (optional)
        Area threshold for object removal.

    Attributes
    ----------
    stack : (T, Y, X) array
        Timelapse data as numpy array.
    r_pool : float
        Radius of pool in pixels.
    d_bw_pool : float
        (Center <--> center) distance between pools in pixels.
    n_pools_max : int
        Max number of pools in timelapse as allowed by geometry.
    Nx, Ny : ints
        Number of pixels in timelapse in X, Y dimensions.
    px_um : float
        x, y pixelsize of timelapse in microns.
    grid : (2, N) array
        indices of pool locations.
    poolmap : dict
        Mapping of grid points to x, y coordinates of pool locations.
    model : `skimage.Transform`
        Transformation to map grid points to pool locations.
    preprocessed : bool
        Whether stack has been preprocessed.
    proj : (Y, X) array
        `projection` intensity projection.
    mask : (Y, X) array
        `projection` intensity projection with edges enhanced for circle detection.
    pools : dict
        Collection of `Pool`s

    Methods
    -------
    load()
    get_n_pools_max()
    preprocess()
    detect_pools()   -->  preprocess()
    extrapoolate()
    find_pools()     -->  detect_pools() + extrapoolate()
    extract_pools()  -->  find_pools()
    export_pools()
    make_debug_sketch()
    """

    def __init__(
            self,
            filepath,
            r_pool_um=50,
            d_bw_pools_um=200,
            t_hough=0.2,
            projection="avg",
            r_median=2,
            r_tophat=5,
            s_rm_objs=500,
        ):

        # check that nd2 file is valid
        self.filepath = filepath
        self._validate()

        # metadata from lazy read
        with nd2.ND2File(filepath) as nd2f:
            voxels_um = nd2f.voxel_size()  # in microns
            sizes = nd2f.sizes  # e.g. {'T': 10, 'C': 2, 'Y': 256, 'X': 256}
        self.Nx = sizes["X"]
        self.Ny = sizes["Y"]
        self.px_um = (voxels_um.x + voxels_um.y) / 2

        # pre-processing parameters
        self.projection = projection
        self.r_median = r_median
        self.r_tophat = r_tophat
        self.s_rm_objs = s_rm_objs
        self.preprocessed = False

        # Hough transform parameters
        self.r_pool = r_pool_um / self.px_um          # um --> px
        self.d_bw_pools = d_bw_pools_um / self.px_um  # um --> px
        self.n_pools_max = self.get_n_pools_max()
        self.t_hough = t_hough

        # initialize empty mapping of grid indices to coordinates of pool locations
        nx = int(np.ceil(self.Nx / self.d_bw_pools))
        ny = int(np.ceil(self.Ny / self.d_bw_pools))
        self.grid = np.mgrid[-1:nx+1, -1:ny+1].reshape(2, -1).T
        self.poolmap = dict.fromkeys([(ix, iy) for (ix, iy) in self.grid])

    def _validate(self):
        """Check that nd2 file has not been corrupted."""
        try:
            with nd2.ND2File(self.filepath) as nd2f:
                nd2f.shape
        except ValueError:
            msg = f"{self.filepath} is corrupted."
            raise ValueError(msg)

    @timeit
    def load(self):
        """Load timelapse from nd2 file as numpy array."""
        if not hasattr(self, "stack"):
            self.stack = nd2.imread(self.filepath)

    def get_n_pools_max(self, generous=True):
        """Calculate max number of pools possible based on geometry."""
        Nx = self.Nx
        Ny = self.Ny
        r = self.r_pool
        d = self.d_bw_pools

        if not generous:
            # kind of a reasonable/expected max
            n_pools_x = int(Nx >= 2*r) + (Nx - 2*r) // d
            n_pools_y = int(Ny >= 2*r) + (Ny - 2*r) // d
        else:
            # truly an upper limit
            n_pools_x = int(np.ceil(Nx / d))
            n_pools_y = int(np.ceil(Ny / d))

        return n_pools_x * n_pools_y

    @timeit
    def preprocess(self):
        """Apply preprocessing steps to a brightfield timelapse of pools image data.

        Process
        -------
        1) Z-projection (in time)
        2) Median filter
        3) Clip intensity range
        4) White tophat filter
        5) Threshold
        6) Clean / remove small segments

        Returns
        -------
        proj : (Y, X) array
            Mean (or whichever) intensity projection image.
        mask :
            Processed image with edges of pools preserved (ideally).
        """
        # create structuring elements (aka footprint) for filters
        footprint_median_filter = ski.morphology.disk(self.r_median)
        footprint_tophat_filter = ski.morphology.disk(self.r_tophat)

        # load stack if not already loaded
        if not hasattr(self, "stack"):
            self.load()

        # do a z-projection (technically a projection in time, but same idea)
        if self.projection.lower() == "avg":
            tproj = self.stack.mean(axis=0)
        elif self.projection.lower() == "max":
            tproj = self.stack.max(axis=0)
        elif self.projection.lower() == "min":
            tproj = self.stack.min(axis=0)
        elif self.projection.lower() == "std":
            tproj = self.stack.std(axis=0)
        elif self.projection.lower() == "med":
            tproj = np.median(self.stack, axis=0)
        else:
            raise ValueError(f"Unknown projection method, {self.projection}.")

        # apply edge-preserving smoothing filter
        image_smooth = ski.filters.median(
            tproj,
            footprint=footprint_median_filter
        )

        # clip intensity range (auto- brightness/contrast)
        med = np.median(image_smooth)
        std = image_smooth.std()
        vmin, vmax = (med - 2*std, med + 2*std)
        image_rescaled = ski.exposure.rescale_intensity(
            image_smooth,
            in_range=(vmin, vmax)
        )
        # tophat filter
        image_pool_edges = ski.morphology.white_tophat(
            image=image_rescaled,
            footprint=footprint_tophat_filter
        )
        # create rough mask on edges of the pools
        thresh = ski.filters.threshold_otsu(image_pool_edges)
        mask = ski.morphology.remove_small_objects(
            image_pool_edges > thresh,
            min_size=self.s_rm_objs
        )

        self.preprocessed = True
        self.proj = image_rescaled
        self.mask = mask

    def detect_pools(self):
        """Run Hough transform on preprocessed timelapse to detect pools.

        This is a first pass at detecting pools using a Hough transform [1]
        to detect circles of a given range of radii.

        References
        ----------
        [1] https://scikit-image.org/docs/stable/auto_examples/edges/plot_circular_elliptical_hough_transform.html
        """
        # preprocess
        if not self.preprocessed:
            self.preprocess()

        # define radii for Hough transform
        r_hough = range(
            round(self.r_pool) - 5,
            round(self.r_pool) + 6,
            2
        )

        # set minimum search distance between adjacent circles
        d_min = int(0.9 * self.d_bw_pools)

        # apply circular Hough transform
        hspaces = ski.transform.hough_circle(
            self.mask,
            r_hough
        )
        hough_peaks = ski.transform.hough_circle_peaks(
            hspaces=hspaces,
            radii=r_hough,
            min_xdistance=d_min,
            min_ydistance=d_min,
            total_num_peaks=self.n_pools_max,
            threshold=self.t_hough
        )

        # unpack Hough peaks
        centers = np.stack([
            hough_peaks[1],  # x coordinates
            hough_peaks[2]   # y coordinates
        ]).T

        return centers

    def extrapoolate(self, centers):
        """Extrapolate pool locations --> "extrapoolate"

        Estimates the coordinates of the center of every potential
        pool in the timelapse based on a least squares fit for a
        Similarity transformation on the (previously) detected
        center coordinates.
        """
        # minimum number of points required to fit a Similarity tranformation
        min_samples = 3
        if len(centers) < min_samples:
            err = (f"Unable to extrapolate pool locations from only "
                   f"{len(centers)} detected pools ({min_samples} needed).")
            raise ValueError(err)

        # define a function for validating the model returned by RANSAC
        def validate_model(model, src, dst):
            """Validate RANSAC model.

            The Similarity transformation for inferring the coordinates of
            the non-detected pools should have a scale approximately equal
            to the distance (in pixels) between each pool. Therefore reject
            all models with scales << or >> than this expected scale.

            Unable to add these parameters to the function call as the function
            is executed inside the call to RANSAC.
            """
            d = self.d_bw_pools
            m = 0.05  # allowed margin of error from expected scale
            return d*(1-m) < model.scale < d*(1+m)

        # bin the coordinates of the detected pools into cells of a grid
        # NOTE: this is unfortunately not very accurate as the grid
        #   of pools is often rotated in the image, but it seems to work
        #   well enough empirically at least as a starting point
        src = centers // self.d_bw_pools

        # use RANSAC to find the linear transformation (w/o shear) that
        # maps the grid indices to the coordinates of the detected pools
        #   (0, 0) --> (118, 95)
        #   (0, 1) --> (145, 399)
        #   (0, 2) --> (172, 703)
        #          ...
        # (nx, ny) --> (cx, cy)
        model, inliers = ski.measure.ransac(
            data=(src, centers),
            model_class=ski.transform.SimilarityTransform,
            min_samples=min_samples,
            residual_threshold=10,
            max_trials=1000,
            is_model_valid=validate_model
        )

        # map grid indices to extrapoolated coordinates with `inferred` status
        # >>> self.poolmap
        #   {(0, 0): [(118, 95), 'inferred'],
        #    (0, 1): [(145, 399), 'inferred'],
        #    (0, 2): [(172, 703), 'inferred']
        #         ...
        #  (nx, ny): [(cx, cy), 'inferred']}
        for (ix, iy) in self.grid:
            cx, cy = model((ix, iy)).ravel().round().astype(int)
            self.poolmap[(ix, iy)] = [(cx, cy), "inferred"]

        self.model = model
        return model, inliers

    @timeit
    def find_pools(self):
        """Detect and extrapoolate pool locations."""

        # detect and extrapolate pool locations
        centers = self.detect_pools()
        model, inliers = self.extrapoolate(centers)
        # NOTE: unfortunately cannot trust inliers here since
        #   mapping of pools to grid locations is not well defined -- same
        #   caveat as mentioned in `extrapoolate` (pools do not have a
        #   predictable/consistent layout from image to image)

        # loop through detected pool locations to update the poolmap
        centers_iT = model.inverse(centers).round()
        for i, (ix, iy) in enumerate(centers_iT):

            # measure the distance between the detected location and the
            # estimated location for outlier detection
            cx, cy = self.poolmap[(ix, iy)][0]
            d = model.residuals(
                (ix, iy),  # detected
                (cx, cy)   # inferred / extrapoolated
            ).item()

            # overwrite extrapoolated locations with detected locations
            status = "inlier" if d < 10 else "outlier"
            self.poolmap[(ix, iy)] = [tuple(centers[i, :]), status]

    @timeit
    def extract_pools(
        self,
        preprocess=True,
        segment=False
    ):
        """Extract pools."""
        # find pools
        self.find_pools()

        # extract pools
        pools = {}
        for (ix, iy), ((cx, cy), _status) in self.poolmap.items():

            # crop to pool (+1 pixel margin)
            try:
                pool_stack = crop_out_prism(
                    stack=self.stack,
                    center=(cx, cy),
                    radius=self.r_pool+1
                )
            # pool extends beyond image border --> skip
            except IndexError:
                continue

            # create pool and optionally process it for cell tracking
            pool = MicrochamberPoolProcessor(pool_stack)
            if preprocess:
                pool.preprocess(remove_stationary_objects=True)
            if segment:
                pool.segment()
            pools[(ix, iy)] = pool

        self.pools = pools
        return pools

    @timeit
    def export_pools(self, dir_out=None):
        """Export processed pools to disk as 8bit tiff stacks.

        Raises
        ------
        AttributeError
            If no pools have been extracted.
        """
        # set default output directory
        if dir_out is None:
            dir_out = self.filepath.parent / "processed"

        # loop through (already processed) pools and save as tiffs
        for (ix, iy), pool in self.pools.items():

            # TODO: determine way to choose which stack from the pool to save
            # convert to 8bit
            pool_8bit = ski.exposure.rescale_intensity(
                pool.stack_prp,
                in_range=(0, 1),
                out_range=(0, 255)
            ).astype(np.ubyte)

            # include pool x, y indices in filename
            tgt = dir_out / self.filepath.stem / f"pool_{ix:02d}_{iy:02d}.tiff"
            tgt.parent.mkdir(exist_ok=True, parents=True)
            ski.io.imsave(tgt, pool_8bit)

    def make_debug_sketch(self, save=True, dir_out=None):
        """Annotates the detected pools for debugging purposes."""

        colormap = {
            "inlier": (20, 255, 255),  # cyan
            "outlier": (255, 150, 20), # orange
            "inferred": (20, 255, 20)  # green
        }

        # convert to 8bit rgb image
        sketch = ski.color.gray2rgb((0.7*255*self.mask).astype(np.ubyte))
        for (_ix, _iy), ((cx, cy), status) in self.poolmap.items():
            # annotate circle outline
            rr, cc = ski.draw.circle_perimeter(
                r=cy,
                c=cx,
                radius=int(self.r_pool),
                shape=sketch.shape
            )
            sketch[rr, cc] = colormap[status]
            # annotate circle center
            rr, cc = ski.draw.disk(
                center=(cy, cx),
                radius=6,
                shape=sketch.shape
            )
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


def crop_out_prism(
    stack,
    center,
    radius,
):
    """Crops a square(ular) prism out of an image stack.

    Parameters
    ----------
    stack : (Z, Y, X) array
        Image stack such as a timelapse or z-stack.
    center : 2-tuple
        (x, y) coordinate -- center of circle from which to crop from.
    radius : scalar
        Radius to determine cropping window (1/2 width of square).

    Returns
    -------
    prism : (Z, Y, X) array
        Cropped image stack with dimensions (Z, 2*R, 2*R).

    Raises
    ------
    IndexError
        If requested crop is outside the extent of the stack.
    """
    # validate input
    cx, cy = tuple(int(i) for i in center)
    r = round(radius)

    # crop to a rectangular prism
    nz, ny, nx = stack.shape
    y1, y2 = cy - r, cy + r
    x1, x2 = cx - r, cx + r
    if (y1 < 0) or (y2 > ny) or (x1 < 0) or (x2 > nx):
        err = (f"Requested crop (array[:, {y1}:{y2}, {x1}:{x2}]) is out of "
               f"bounds for array with shape {stack.shape}.")
        raise IndexError(err)
    else:
        prism = stack[:, y1:y2, x1:x2]

    return prism
