import click
import numpy as np
import skimage as ski
from chlamytracker import cli_options
from chlamytracker.pool_finder import PoolFinder
from chlamytracker.tracking import PoolTracker
from natsort import natsorted
from tqdm import tqdm


def process_timelapse(filename, pool_radius_um, pool_spacing_um, btrack_config_file):
    """Function for processing an individual file of (raw) timelapse microscopy
    data of unicellular organisms in agar microchamber pools."""
    # extract, segment, and export pools
    pool_finder = PoolFinder(
        filepath=filename,
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
            pool_tracker = PoolTracker(segmentation_data, btrack_config_file)
            pool_tracker.track_cells()
            csv_filename = (
                pool_finder.filepath.parent
                / "processed"
                / pool_finder.filepath.stem
                / f"pool_{ix:02d}_{iy:02d}_tracks.csv"
            )
            pool_tracker.tracker.export(csv_filename)


@cli_options.btrack_config_file_option
@cli_options.pool_spacing_um_option
@cli_options.pool_radius_um_option
@cli_options.data_dir_option
@click.command()
def main(data_dir, pool_radius_um, pool_spacing_um, btrack_config_file):
    """Script for batch processing (raw) timelapse microscopy data of
    unicellular organisms in agar microchamber pools.

    For every .nd2 file in the given directory, this script will output a .csv
    file of motility data (x, y position) and object properties (e.g. area,
    eccentricity) of segmented cells for each frame in the timelapse.
    """

    # glob all .nd2 files in directory
    fps_nd2 = natsorted(data_dir.glob("*.nd2"))
    if not fps_nd2:
        raise ValueError(f"No .nd2 files found in {data_dir}")

    # loop through .nd2 files
    for fp in tqdm(fps_nd2):
        try:
            process_timelapse(fp, pool_radius_um, pool_spacing_um, btrack_config_file)

        # skip over failures caused by
        # > corrupt nd2 files
        # > when extrapolation fails due to < 3 detected pools
        # > segmentation failures
        except ValueError as err:
            msg = f"Processing for {fp} failed:"
            print(msg, err)


if __name__ == "__main__":
    main()
