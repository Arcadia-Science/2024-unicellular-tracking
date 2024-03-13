import click
import skimage as ski
from chlamytracker import cli_options
from chlamytracker.tracking import PoolTracker
from natsort import natsorted
from tqdm import tqdm


@cli_options.data_dir_option
@cli_options.btrack_config_file_option
@click.command()
def main(data_dir, btrack_config_file):
    """"""

    # glob all the tiff files in directory
    fps_tiff = natsorted(data_dir.glob("processed/*/pool*.tiff"))
    if not fps_tiff:
        raise ValueError(f"No .tiff files found in {data_dir}")

    # loop through tiff files for tracking
    for fp in tqdm(fps_tiff):

        # load segmentation
        segmentation_data = ski.io.imread(fp)

        # track cells
        pool_tracker = PoolTracker(segmentation_data, btrack_config_file)
        pool_tracker.track_cells()

        # export tracking data
        tgt = fp.parent / f"{fp.stem}_tracks.csv"
        pool_tracker.tracker.export(tgt)


if __name__ == "__main__":
    main()
