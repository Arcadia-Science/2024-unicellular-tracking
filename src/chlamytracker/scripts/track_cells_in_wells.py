import click
from chlamytracker import cli_options
from chlamytracker.tracking import Tracker
from chlamytracker.well_processor import WellProcessor
from natsort import natsorted
from tqdm import tqdm


def process_timelapse(filename, btrack_config_file):
    """Function for processing an individual file of raw timelapse microscopy
    data of unicellular organisms in agar microchamber pools."""
    # segmentation
    well_processor = WellProcessor(filename)
    segmentation = well_processor.segment()

    # cell tracking
    well_tracker = Tracker(segmentation, btrack_config_file)
    well_tracker.track_cells()

    # export tracking data
    csv_filename = (
        well_processor.nd2_file.parent / "processed" / f"{well_processor.nd2_file.stem}_tracks.csv"
    )
    csv_filename.parent.mkdir(exist_ok=True, parents=False)
    well_tracker.tracker.export(csv_filename)


@cli_options.btrack_config_file_option
@cli_options.data_dir_option
@click.command()
def main(data_dir, btrack_config_file):
    """Script for batch processing raw timelapse microscopy data of
    unicellular organisms in 384 or 1536 well plates.

    For every .nd2 file in the given directory, this script will output a .csv
    file of motility data (i.e. (x, y) positions) and object properties (e.g.
    area, eccentricity) of segmented cells for each frame in the timelapse.
    """

    # glob all .nd2 files in directory
    nd2_files = natsorted(data_dir.glob("*.nd2"))
    if not nd2_files:
        raise ValueError(f"No .nd2 files found in {data_dir}")

    # loop through .nd2 files
    for nd2_file in tqdm(nd2_files):
        try:
            process_timelapse(nd2_file, btrack_config_file)

        # skip over failures caused by
        #   - corrupt nd2 files
        #   - segmentation failures
        except ValueError as err:
            msg = f"Processing for {nd2_file} failed:"
            print(msg, err)


if __name__ == "__main__":
    main()
