import logging

import click
import numpy as np
import skimage as ski
from swimtracker import cli_options
from swimtracker.pool_finder import PoolFinder
from swimtracker.tracking import Tracker
from swimtracker.well_processor import WellSegmenter
from natsort import natsorted
from tqdm import tqdm

logger = logging.getLogger(__name__)


def process_timelapse_of_well(
    nd2_file,
    output_directory,
    min_cell_diameter_um,
    num_workers,
    use_dask,
    btrack_config_file,
    verbose,
):
    """Function for processing an individual file of raw timelapse microscopy
    data of unicellular organisms in a 384-well or 1536-well plate."""

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
    dataframe = well_tracker.to_dataframe()
    dataframe.to_csv(csv_filename, index=False)


def process_timelapse_of_pools(
    nd2_file,
    output_directory,
    min_cell_diameter_um,
    pool_radius_um,
    pool_spacing_um,
    num_workers,
    btrack_config_file,
    verbose,
):
    """Function for processing an individual nd2 file of raw timelapse microscopy
    data of unicellular organisms in agar microchamber pools."""
    # find pools within the timelapse
    pool_finder = PoolFinder(
        nd2_file=nd2_file,
        pool_radius_um=pool_radius_um,
        pool_spacing_um=pool_spacing_um,
        min_cell_diameter_um=min_cell_diameter_um,
    )

    # configure export directory
    #   output segmentation and tracking data to subdirectories as there are
    #   multiple pools per nd2 file
    output_directory /= pool_finder.nd2_file.stem
    output_directory.mkdir(exist_ok=True)

    # segment cells within each pool
    pools_segmented = pool_finder.segment_pools()

    # export poolmap
    txt_file = output_directory / "poolmap.txt"
    with open(txt_file, "w") as txt:
        for (ix, iy), ((cx, cy), status) in pool_finder.poolmap.items():
            line = f"{ix}\t{iy}\t{cx}\t{cy}\t{status}\n"
            txt.write(line)

    # render and save debug sketch of detected pools
    pools_debug_sketch = pool_finder.make_debug_sketch()
    jpg_file = output_directory / "pools_detected.jpg"
    ski.io.imsave(jpg_file, pools_debug_sketch)

    # track segmented cells in each pool and output to csv
    for (ix, iy), segmentation in pools_segmented.items():
        # export segmentation
        tiff_file = output_directory / f"pool_{ix}_{iy}_segmented.tiff"
        segmentation_8bit = (255 * segmentation).astype(np.uint8)
        ski.io.imsave(tiff_file, segmentation_8bit)

        # cell tracking
        pool_tracker = Tracker(segmentation_8bit, btrack_config_file, num_workers, verbose)
        pool_tracker.track_cells()

        # export tracking data
        csv_file = output_directory / f"pool_{ix}_{iy}_tracks.csv"
        pool_tracker.tracker.export(csv_file)


@click.command()
@cli_options.input_directory_argument
@cli_options.output_directory_option
@cli_options.glob_option
@cli_options.vessel_type_option
@cli_options.min_cell_diameter_um_option
@cli_options.pool_radius_um_option
@cli_options.pool_spacing_um_option
@cli_options.num_workers_option
@cli_options.use_dask_option
@cli_options.btrack_config_file_option
@cli_options.verbose_option
def main(
    input_directory,
    output_directory,
    glob_str,
    vessel_type,
    min_cell_diameter_um,
    pool_radius_um,
    pool_spacing_um,
    num_workers,
    use_dask,
    btrack_config_file,
    verbose,
):
    """Script for batch processing raw timelapse microscopy data of unicellular
    organisms in 384 or 1536 well plates or agar microchamber pools [1].

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

    Notes
    -----
    * Results are output to `{input_directory}/processed` by default if
    `output_directory` is not specified.
    * `num_workers` option is ignored when the `use_dask` is provided since dask
    pretty much uses all available computing power at its disposal.

    References
    ----------
    [1] https://doi.org/10.57844/arcadia-v1bg-6b60
    [2] https://btrack.readthedocs.io/en/latest/index.html
    """

    # set log level
    if verbose:
        logger.setLevel(logging.DEBUG)

    # glob all .nd2 files in directory
    nd2_files = natsorted(input_directory.glob(glob_str))
    if not nd2_files:
        logger.error(f"No nd2 files found in {input_directory}.")

    # ensure output directory exists and is writeable
    if output_directory is None:
        output_directory = input_directory / "processed"
    output_directory.mkdir(parents=True, exist_ok=True)

    if "well" in vessel_type.lower():
        # loop through nd2 files
        for nd2_file in tqdm(nd2_files):
            try:
                process_timelapse_of_well(
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

    elif "pool" in vessel_type.lower():
        # loop through nd2 files
        for nd2_file in tqdm(nd2_files):
            try:
                process_timelapse_of_pools(
                    nd2_file,
                    output_directory,
                    min_cell_diameter_um,
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
