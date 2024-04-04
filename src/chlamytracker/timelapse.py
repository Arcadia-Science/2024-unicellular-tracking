import logging
import math
from pathlib import Path

import nd2

from .utils import timeit

logger = logging.getLogger(__name__)


class Timelapse:
    """Class for processing timelapse microscopy data.

    Currently only supports the use of nd2 file format.

    Parameters
    ----------
    nd2_file : Path | str
        Filepath to input nd2 file.
    use_dask : bool
        Whether to read nd2 file with dask.
    """

    def __init__(
        self,
        nd2_file,
        use_dask=False,
    ):

        self.nd2_file = Path(nd2_file)
        self.use_dask = use_dask

        # check that the nd2 file is not corrupted
        self.is_corrupted = self._validate_nd2_file()

        # metadata from nd2 headers
        with nd2.ND2File(nd2_file) as nd2f:
            voxels_um = nd2f.voxel_size()  # in microns
            self.dimensions = nd2f.sizes  # e.g. {'T': 10, 'C': 2, 'Y': 256, 'X': 256}
        self.um_per_px = (voxels_um.x + voxels_um.y) / 2

        # determine whether timelapse is also a zstack
        self.is_zstack = self.dimensions.get("Z") is not None

        # load data from nd2 file
        logger.info(f"Loading timelapse {self.nd2_file.name} with dimensions: {self.dimensions}")
        self.raw_data = self.load()

    def _validate_nd2_file(self):
        """Check that nd2 file has not been corrupted."""
        try:
            # simply checking for shape will determine if nd2 file is corrupted
            with nd2.ND2File(self.nd2_file) as nd2f:
                _ = nd2f.shape
            return False

        except ValueError as err:
            msg = f"{self.nd2_file} is corrupted."
            logger.error(msg, exc_info=err)
            return True

    @timeit
    def load(self):
        """Load timelapse from nd2 file as numpy array."""
        # check that nd2 file has not already been loaded
        if hasattr(self, "raw_data"):
            logger.warn(f"{self.nd2_file.as_posix()} has already been loaded.")
            return

        raw_data = nd2.imread(self.nd2_file, dask=self.use_dask)
        return raw_data


    def convert_um_to_px2_circle(self, diameter_um):
        """Convert diameter [um] --> area [px^2]."""
        radius_um = diameter_um / 2
        radius_px = radius_um / self.um_per_px
        area_px2 = math.pi * radius_px**2
        return round(area_px2)
