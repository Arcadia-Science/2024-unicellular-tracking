import logging
from pathlib import Path

import nd2

from .utils import timeit

logger = logging.getLogger(__name__)


class Timelapse:
    """Class for processing timelapse microscopy data.

    Parameters
    ----------
    nd2_file : Path | str
        Filepath to input nd2 file.
    """
    def __init__(
        self,
        nd2_file,
    ):
        self.nd2_file = Path(nd2_file)

        # check that the nd2 file is not corrupted
        self.is_corrupted = self._validate_nd2_file()

        # metadata from nd2 headers
        with nd2.ND2File(nd2_file) as nd2f:
            voxels_um = nd2f.voxel_size()  # in microns
            self.dimensions = nd2f.sizes  # e.g. {'T': 10, 'C': 2, 'Y': 256, 'X': 256}
        self.um_per_px = (voxels_um.x + voxels_um.y) / 2

        # load data from nd2 file
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
        if not hasattr(self, "raw_data"):
            raw_data = nd2.imread(self.nd2_file)
            return raw_data
