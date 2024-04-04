import logging
from pathlib import Path

import click
import numpy as np
import skimage as ski
from chlamytracker import cli_api
from chlamytracker.tracking import Tracker
from chlamytracker.well_processor import WellSegmenter
from natsort import natsorted
from tqdm import tqdm

logger = logging.getLogger(__name__)

EXPORT_ROOT_DIRECTORY = Path("/home/theia/arc_nas_mirror/")


def process_timelapse(filename, min_cell_diameter_um, btrack_config_file):
    """Function for processing an individual file of raw timelapse microscopy
    data of unicellular organisms in agar microchamber pools."""

    # segmentation
    well_processor = WellSegmenter(filename)
    segmentation = well_processor.segment(min_cell_diameter_um)

    # configure export directory
    stripped_directory_tree = well_processor.nd2_file.parts
    index_home = stripped_directory_tree.index("home")
    stem_directory = Path(*stripped_directory_tree[index_home + 3 : -1]) / "processed"
    export_directory = EXPORT_ROOT_DIRECTORY / stem_directory
    export_directory.mkdir(parents=True, exist_ok=True)
    print(export_directory)

    # export segmentation
    tiff_filename = export_directory / f"{well_processor.nd2_file.stem}_segmented.tiff"
    segmentation_8bit = (255 * segmentation).astype(np.uint8)
    ski.io.imsave(tiff_filename, segmentation_8bit)

    # cell tracking
    well_tracker = Tracker(segmentation_8bit, btrack_config_file)
    well_tracker.track_cells()

    # export tracking data
    csv_filename = export_directory / f"{well_processor.nd2_file.stem}_tracks.csv"
    well_tracker.tracker.export(csv_filename)


@cli_api.btrack_config_file_option
@cli_api.min_cell_diameter_um_option
@cli_api.glob_option
@cli_api.directory_argument
@click.command()
def main(directory, glob_str, min_cell_diameter_um, btrack_config_file):
    """Script for batch processing raw timelapse microscopy data of unicellular
    organisms in 384 or 1536 well plates.

    For every nd2 file returned by the glob search pattern, this script will
    perform segmentation and cell tracking and will output the results to
    `{directory}/processed`. A tiff file is output for the segmentation and a
    csv file of motility data is output for the cell tracking that contains the
    (x, y) position and object properties (e.g. area, eccentricity, etc.) of
    each tracked cell for each frame in the timelapse.
    """

    # glob all .nd2 files in directory
    nd2_files = natsorted(directory.glob(glob_str))
    if not nd2_files:
        raise ValueError(f"No nd2 files found in {directory}.")

    # loop through nd2 files
    for nd2_file in tqdm(nd2_files):
        try:
            process_timelapse(nd2_file, min_cell_diameter_um, btrack_config_file)

        # skip over failures caused by
        #   - corrupt nd2 files
        #   - segmentation failures
        except ValueError as err:
            msg = f"Processing for {nd2_file} failed:"
            print(msg, err)


if __name__ == "__main__":
    main()
