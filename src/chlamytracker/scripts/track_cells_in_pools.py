import logging

import click
import numpy as np
import skimage as ski
from chlamytracker import cli_api
from chlamytracker.pool_finder import PoolFinder
from chlamytracker.tracking import Tracker
from natsort import natsorted
from tqdm import tqdm

logger = logging.getLogger(__name__)


def process_timelapse(
    nd2_file,
    output_directory,
    pool_radius_um,
    pool_spacing_um,
    num_workers,
    btrack_config_file,
    verbose,
):
    """Function for processing an individual file of raw timelapse microscopy
    data of unicellular organisms in agar microchamber pools."""
    # extract, segment, and export pools
    pool_finder = PoolFinder(
        filepath=nd2_file,
        pool_radius_um=pool_radius_um,
        pool_spacing_um=pool_spacing_um,
    )
    pool_finder.extract_pools()
    pool_finder.make_debug_sketch()
    pool_finder.segment_pools()
    pool_finder.export_pools()

    # track segmented cells in each pool and output to csv
    for (ix, iy), pool in pool_finder.pools.items():
        if pool.has_cells() and pool.is_segmented:
            # rescale segmentation data for btrack
            segmentation_data = ski.exposure.rescale_intensity(
                pool.stack_segmented, out_range=(0, 255)
            ).astype(np.uint8)

            # cell tracking
            pool_tracker = Tracker(segmentation_8bit, btrack_config_file, num_workers, verbose)
            pool_tracker.track_cells()
            csv_filename = (
                pool_finder.filepath.parent
                / "processed"
                / pool_finder.filepath.stem
                / f"pool_{ix:02d}_{iy:02d}_tracks.csv"
            )
            pool_tracker.tracker.export(csv_filename)


@click.command()
@cli_api.input_directory_argument
@cli_api.output_directory_option
@cli_api.glob_option
@cli_api.pool_radius_um_option
@cli_api.pool_spacing_um_option
@cli_api.num_workers_option
@cli_api.btrack_config_file_option
@cli_api.verbose_option
def main(
    input_directory,
    output_directory,
    glob_str,
    pool_radius_um,
    pool_spacing_um,
    num_workers,
    btrack_config_file,
    verbose,
):
    """Script for batch processing raw timelapse microscopy data of unicellular
    organisms in agar microchambers [1].

    This script performs segmentation and cell tracking on each nd2 file
    returned by the glob search pattern. Cell tracking will only proceed if
    the segmentation was successful. If the segmentation fails (most likely
    due to poor thresholding as a result of poor image quality or the absence
    of cells), the nd2 file is skipped and nothing is output. If the
    segmentation succeeds, a tiff file of the segmented timelapse is output and
    cell tracking of the segmented timelapse will start. Cell tracking is done
    using `btrack` [2]. Assuming cell tracking completes successfully, a csv
    file of motility data is output that contains the (x, y) position and object
    properties (e.g. area, eccentricity, etc.) of each tracked cell for each
    frame in the timelapse.

    Results are output to `{input_directory}/processed` by default if
    `output_directory` is not specified.

    References
    ----------
    [1] https://doi.org/10.57844/arcadia-v1bg-6b60
    [2] https://btrack.readthedocs.io/en/latest/index.html
    """

    # set log level
    if verbose:
        logger.setLevel(logging.INFO)

    # glob all .nd2 files in directory
    nd2_files = natsorted(input_directory.glob(glob_str))
    if not nd2_files:
        logger.error(f"No nd2 files found in {input_directory}.")

    # ensure output directory exists and is writeable
    if output_directory is None:
        output_directory = input_directory / "processed"
    output_directory.mkdir(parents=True, exist_ok=True)

    # loop through nd2 files
    for nd2_file in tqdm(nd2_files):
        try:
            process_timelapse(
                nd2_file,
                output_directory,
                pool_radius_um,
                pool_spacing_um,
                num_workers,
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
