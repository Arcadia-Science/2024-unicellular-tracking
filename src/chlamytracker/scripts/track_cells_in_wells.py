import logging

import click
import numpy as np
import skimage as ski
from chlamytracker import cli_api
from chlamytracker.tracking import Tracker
from chlamytracker.well_processor import WellSegmenter
from natsort import natsorted
from tqdm import tqdm

logger = logging.getLogger(__name__)


def process_timelapse(
    nd2_file,
    output_directory,
    min_cell_diameter_um,
    num_workers,
    use_dask,
    btrack_config_file,
    verbose,
):
    """Function for processing an individual file of raw timelapse microscopy
    data of unicellular organisms in agar microchamber pools."""

    # segmentation
    well = WellSegmenter(nd2_file, use_dask=use_dask)
    segmentation = well.segment(min_cell_diameter_um)

    # export segmentation
    tiff_filename = output_directory / f"{well.nd2_file.stem}_segmented.tiff"
    segmentation_8bit = (255 * segmentation).astype(np.uint8)
    ski.io.imsave(tiff_filename, segmentation_8bit)

    # cell tracking
    well_tracker = Tracker(segmentation_8bit, btrack_config_file, num_workers, verbose)
    well_tracker.track_cells()

    # export tracking data
    csv_filename = output_directory / f"{well.nd2_file.stem}_tracks.csv"
    well_tracker.tracker.export(csv_filename)


@click.command()
@cli_api.verbose_option
@cli_api.btrack_config_file_option
@cli_api.use_dask_option
@cli_api.num_workers_option
@cli_api.min_cell_diameter_um_option
@cli_api.glob_option
@cli_api.output_directory_option
@cli_api.input_directory_argument
def main(
    input_directory,
    output_directory,
    glob_str,
    min_cell_diameter_um,
    num_workers,
    use_dask,
    btrack_config_file,
    verbose,
):
    """Script for batch processing raw timelapse microscopy data of unicellular
    organisms in 384 or 1536 well plates.

    This script performs segmentation and cell tracking on each nd2 file
    returned by the glob search pattern. Cell tracking will only proceed if
    the segmentation was successful. If the segmentation fails (most likely
    due to poor thresholding as a result of poor image quality or the absence
    of cells), the nd2 file is skipped and nothing is output. If the
    segmentation succeeds, a tiff file of the segmented timelapse is output and
    cell tracking of the segmented timelapse will start. Cell tracking is done
    using `btrack` [1]. Assuming cell tracking completes successfully, a csv
    file of motility data is output that contains the (x, y) position and object
    properties (e.g. area, eccentricity, etc.) of each tracked cell for each
    frame in the timelapse.

    Results are output to `{input_directory}/processed` by default if
    `output_directory` is not specified.

    `num_workers` option is ignored if `use_dask` is True since dask pretty much
    uses all available computing power at its disposal.

    References
    ----------
    [1] https://btrack.readthedocs.io/en/latest/index.html
    """

    # set log level
    if verbose:
        logger.setLevel(logging.INFO)

    # glob all .nd2 files in directory
    nd2_files = natsorted(input_directory.glob(glob_str))
    if not nd2_files:
        raise ValueError(f"No nd2 files found in {input_directory}.")

    # ensure output directory exists and is writeable
    output_directory.mkdir(parents=True, exist_ok=True)

    # loop through nd2 files
    for nd2_file in tqdm(nd2_files):
        try:
            process_timelapse(
                nd2_file,
                output_directory,
                min_cell_diameter_um,
                num_workers,
                use_dask,
                btrack_config_file,
                verbose,
            )

        # skip over segmentation failures and corrupt nd2 files
        except ValueError as err:
            msg = f"Processing for {nd2_file} failed:"
            logger.warning(msg + str(err))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
