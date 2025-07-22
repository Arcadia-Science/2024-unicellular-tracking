from __future__ import annotations
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dask.array as da
import nd2
import numpy as np
import skimage as ski
from dask import delayed
from numpy.typing import NDArray
from tifffile import TiffFile

logger = logging.getLogger(__name__)


@dataclass
class Timelapse:
    """A timelapse microscopy data container.

    Supports loading from ND2 and TIFF formats, with metadata extraction
    and temporal information processing.

    Attributes:
        raw_data: The image data array.
        dimensions: Dictionary mapping dimensions to sizes (e.g., {'T': 100, 'Y': 512, 'X': 512}).
        pixelsize_um: Pixel size in micrometers.
        frametime_s: Time between frames in seconds.
        use_dask: Whether to use dask for lazy loading (optional).
        metadata: Additional metadata from the image file (optional).
    """

    raw_data: NDArray
    dimensions: dict[str, int]
    pixelsize_um: float
    frametime_s: float

    use_dask: bool = False
    metadata: dict[str, Any] | nd2.structures.Metadata | None = None

    def __post_init__(self):
        """Initialize computed attributes after dataclass creation."""
        if self.frametime_s <= 0:
            raise ValueError(f"Frame time must be positive, got {self.frametime_s}")
        self.framerate_hz = 1 / self.frametime_s

    @classmethod
    def from_nd2_path(
        cls,
        nd2_path: Path | str,
        use_dask: bool = False,
        metadata: dict[str, Any] | nd2.structures.Metadata | None = None,
    ) -> Timelapse:
        """Create a Timelapse from an ND2 file.

        Args:
            nd2_path: Path to the ND2 file.
            use_dask: Whether to use dask for lazy loading.
            metadata: Optional metadata override.

        Returns:
            Timelapse instance loaded from the ND2 file.

        Raises:
            ValueError: If the file has no time dimension or is corrupted.
            FileNotFoundError: If the ND2 file doesn't exist.
        """

        if not nd2_path.exists():
            raise FileNotFoundError(f"ND2 file not found: {nd2_path}")

        # Parse ND2 metadata
        try:
            with nd2.ND2File(nd2_path) as nd2f:
                if metadata is None:
                    metadata = nd2f.metadata
                voxels_um = nd2f.voxel_size()
                sizes = nd2f.sizes  # e.g. {'T': 10, 'C': 2, 'Y': 256, 'X': 256}
                events = nd2f.events()
        except Exception as e:
            raise ValueError(f"Failed to read ND2 file {nd2_path}: {e}") from e

        # Check for time dimension
        if "T" not in sizes:
            raise ValueError(f"Input ND2 file {nd2_path} has no time dimension.")
        # Extract timing info
        if not events:
            raise ValueError(f"No timing events found in ND2 file {nd2_path}")
        frametimes_s = np.diff([event["Time [s]"] for event in events])
        if len(frametimes_s) == 0:
            frametime_s = 1.0  # Default to 1 second if only one frame
            logger.warning(f"Only one frame found in {nd2_path}, defaulting to 1s frame time.")
        else:
            frametime_s = frametimes_s.mean()
            # Warn if frame times are inconsistent
            if frametimes_s.std() / frametime_s > 0.01:
                logger.warning(
                    f"Inconsistent frame times in {nd2_path.name}: "
                    f"{1e3 * frametime_s:.1f} ± {1e3 * frametimes_s.std():.2f} ms"
                )

        # Extract spatial info
        pixelsize_um = (voxels_um.x + voxels_um.y) / 2

        # Load image data
        try:
            raw_data = nd2.imread(nd2_path, dask=use_dask)
        except Exception as e:
            raise ValueError(f"Failed to load image data from ND2 file {nd2_path}: {e}") from e
        logger.info(f"Loaded ND2 {nd2_path.name} with dimensions: {sizes}")

        return cls(
            raw_data,
            sizes,
            pixelsize_um,
            frametime_s,
            use_dask,
            metadata,
        )

    @classmethod
    def from_tiff_path(
        cls,
        tiff_path: Path | str,
        pixelsize_um: float,
        frametime_s: float,
        use_dask: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Timelapse:
        """Create a Timelapse from a TIFF file.

        Args:
            tiff_path: Path to the TIFF file.
            pixelsize_um: Pixel size in micrometers (must be provided for TIFF).
            frametime_s: Time between frames in seconds (must be provided for TIFF).
            use_dask: Whether to use dask for lazy loading.
            metadata: Optional metadata.

        Returns:
            Timelapse instance loaded from the TIFF file.

        Raises:
            ValueError: If the file has no time dimension or parameters are invalid.
            FileNotFoundError: If the TIFF file doesn't exist.
        """
        if not tiff_path.exists():
            raise FileNotFoundError(f"TIFF file not found: {tiff_path}")

        if pixelsize_um <= 0:
            raise ValueError(f"Pixel size must be positive, got {pixelsize_um}")

        if frametime_s <= 0:
            raise ValueError(f"Frame time must be positive, got {frametime_s}")

        # Extract what info we can from a generic TIFF file
        try:
            with TiffFile(tiff_path) as tiff:
                series = tiff.series[0]
                if len(series.shape) != 3:
                    raise ValueError(
                        f"Image data must have 3 dimensions, found {len(series.shape)}"
                    )
                if metadata is None:
                    metadata = tiff.pages[0].description
        except Exception as e:
            raise ValueError(f"Failed to read TIFF file {tiff_path}: {e}") from e

        # Provide mapping for dimensions
        dimensions = {k: v for k, v in zip(series.axes, series.shape, strict=True)}

        # Load image data
        if use_dask:
            raw_data = lazy_load_tiff(tiff_path)
        else:
            raw_data = ski.io.imread(tiff_path)
        logger.info(f"Loaded TIFF {tiff_path.name} with dimensions: {dimensions}")

        return cls(
            raw_data,
            dimensions,
            pixelsize_um,
            frametime_s,
            use_dask,
            metadata,
        )

    @property
    def num_frames(self) -> int:
        """Number of time frames in the timelapse."""
        return self.dimensions.get("T", 1)

    @property
    def duration_s(self) -> float:
        """Total duration of the timelapse in seconds."""
        return (self.num_frames - 1) * self.frametime_s if self.num_frames > 1 else 0.0

    def convert_um_to_px2_circle(self, diameter_um: float) -> int:
        """Convert diameter in micrometers to area in square pixels.

        Args:
            diameter_um: Diameter in micrometers

        Returns:
            Area in square pixels (rounded to nearest integer)
        """
        if diameter_um <= 0:
            raise ValueError(f"Diameter must be positive, got {diameter_um}")

        radius_um = diameter_um / 2
        radius_px = radius_um / self.pixelsize_um
        area_px2 = math.pi * radius_px**2
        return round(area_px2)


def lazy_load_tiff(tiff_path: Path | str) -> da.Array:
    """Load a TIFF file lazily using dask.

    Args:
        tiff_path: Path to the TIFF file to load.

    Returns:
        Dask array containing the image data.

    Note:
        This function creates a delayed read operation that will be executed
        only when the data is actually accessed.
    """

    def read_tiff(tiff_path):
        with TiffFile(tiff_path) as tiff:
            return tiff.asarray()

    with TiffFile(tiff_path) as tiff:
        series = tiff.series[0]
        shape = series.shape
        dtype = series.dtype

    lazy_data = delayed(read_tiff)(tiff_path)
    stack = da.stack(da.from_delayed(lazy_data, shape, dtype))
    return stack
