from __future__ import annotations
import logging
from pathlib import Path

import click
import numpy as np
import skimage as ski
from natsort import natsorted
from swimtracker import cli_options
from swimtracker.timelapse import Timelapse
from swimtracker.tracking import Tracker
from swimtracker.well_processor import WellSegmenter
from tqdm import tqdm

logger = logging.getLogger(__name__)


def create_timelapse(
    input_path: Path | str,
    pixelsize_um: float | None = None,
    frametime_s: float | None = None,
    use_dask: bool = False,
) -> Timelapse:
    """Create a Timelapse object from input file.

    Args:
        input_path: Path to the input file (TIFF or ND2).
        pixelsize_um: Pixel size in micrometers (required for TIFF files).
        frametime_s: Frame time in seconds (required for TIFF files).
        use_dask: Whether to use dask for lazy loading.

    Returns:
        Timelapse object loaded from the input file.

    Raises:
        ValueError: If file type is not supported or required parameters are missing.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() in [".tif", ".tiff"]:
        if pixelsize_um is None or frametime_s is None:
            raise ValueError("TIFF files require pixelsize_um and frametime_s parameters.")
        timelapse = Timelapse.from_tiff_path(input_path, pixelsize_um, frametime_s, use_dask)
    elif input_path.suffix.lower() == ".nd2":
        timelapse = Timelapse.from_nd2_path(input_path, use_dask)
    else:
        raise ValueError(f"Unsupported file type: {input_path.suffix}. Must be TIFF or ND2.")

    return timelapse


def process_timelapse_of_well(
    input_path: Path,
    output_directory: Path,
    min_cell_diameter_um: float,
    pixelsize_um: float,
    frametime_s: float,
    num_workers: int,
    use_dask: bool,
    btrack_config_file: Path | None,
    verbose: bool,
) -> None:
    """Process timelapse data from a single well.

    Performs segmentation and cell tracking on microscopy data from individual
    wells of multi-well plates.

    Args:
        input_path: Path to the input timelapse file.
        output_directory: Directory to save output files.
        min_cell_diameter_um: Minimum cell diameter in micrometers for filtering.
        pixelsize_um: Pixel size in micrometers (required for TIFF files).
        frametime_s: Frame time in seconds (required for TIFF files).
        num_workers: Number of parallel workers to use.
        use_dask: Whether to use dask for processing.
        btrack_config_file: Path to btrack configuration file.
        verbose: Whether to enable verbose logging.

    Raises:
        ValueError: If segmentation fails or file processing encounters errors.
    """

    # segmentation
    timelapse = create_timelapse(input_path, pixelsize_um, frametime_s, use_dask)
    segmenter = WellSegmenter(timelapse)
    segmentation = segmenter.segment(
        min_cell_diameter_um=min_cell_diameter_um, num_workers=num_workers
    )

    # export segmentation
    tiff_filename = output_directory / f"{input_path.stem}_segmented.tiff"
    segmentation_8bit = (255 * segmentation).astype(np.uint8)
    ski.io.imsave(tiff_filename, segmentation_8bit)

    # cell tracking
    well_tracker = Tracker(segmentation_8bit, btrack_config_file, num_workers, verbose)
    well_tracker.track_cells()

    # export tracking data
    csv_filename = output_directory / f"{input_path.stem}_tracks.csv"
    dataframe = well_tracker.to_dataframe()
    dataframe.to_csv(csv_filename, index=False)


@click.command()
@cli_options.input_directory_argument
@cli_options.output_directory_option
@cli_options.glob_option
@cli_options.min_cell_diameter_um_option
@cli_options.pixelsize_um_option
@cli_options.frametime_s_option
@cli_options.num_workers_option
@cli_options.use_dask_option
@cli_options.btrack_config_file_option
@cli_options.verbose_option
def main(
    input_directory: Path,
    output_directory: Path | None,
    glob_str: str,
    min_cell_diameter_um: float,
    pixelsize_um: float,
    frametime_s: float,
    num_workers: int,
    use_dask: bool,
    btrack_config_file: Path | None,
    verbose: bool,
) -> None:
    """Script for batch processing raw timelapse microscopy data of unicellular
    organisms in multi-well plates [1].

    This script performs segmentation and cell tracking on each timelapse file
    (ND2 or TIFF) returned by the glob search pattern. Cell tracking will only
    proceed if the segmentation was successful. If the segmentation fails (most
    likely due to poor thresholding as a result of poor image quality or the
    absence of cells), the file is skipped and nothing is output. If the
    segmentation succeeds, a TIFF file of the segmented timelapse is output and
    cell tracking of the segmented timelapse will start. Cell tracking is done
    using `btrack` [2]. Assuming cell tracking completes successfully, a CSV
    file of motility data is output that contains the (x, y) position and object
    properties (e.g. area, eccentricity, etc.) of each tracked cell for each
    frame in the timelapse.

    Notes:
        * Results are output to `{input_directory}/processed` by default if
          `output_directory` is not specified.
        * `num_workers` option is ignored when the `use_dask` is provided since dask
          pretty much uses all available computing power at its disposal.

    References:
        [1] https://doi.org/10.57844/arcadia-v1bg-6b60
        [2] https://btrack.readthedocs.io/en/latest/index.html
    """

    if verbose:
        logger.setLevel(logging.DEBUG)

    # Glob all files in directory
    input_paths = natsorted(input_directory.glob(glob_str))
    if not input_paths:
        logger.error(f"No files found matching '{glob_str}' in {input_directory}.")
        return

    # Ensure output directory exists and is writeable
    if output_directory is None:
        output_directory = input_directory / "processed"
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create output directory {output_directory}: {e}")
        return

    # Loop through timelapse files
    for input_path in tqdm(input_paths):
        try:
            process_timelapse_of_well(
                input_path,
                output_directory,
                min_cell_diameter_um,
                pixelsize_um,
                frametime_s,
                num_workers,
                use_dask,
                btrack_config_file,
                verbose,
            )

        # Skip over segmentation failures and corrupt files
        except (ValueError, FileNotFoundError, OSError) as err:
            logger.warning(f"Processing for {input_path} failed: {err}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
