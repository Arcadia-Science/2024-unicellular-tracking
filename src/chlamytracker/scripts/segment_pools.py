import click
from chlamytracker import cli_options
from chlamytracker.pool_finder import PoolFinder
from natsort import natsorted
from tqdm import tqdm


@cli_options.pool_spacing_um_option
@cli_options.pool_radius_um_option
@cli_options.data_dir_option
@click.command()
def main(data_dir, pool_radius_um, pool_spacing_um):
    """Wrapper for PoolFinder.extract_pools() and PoolFinder.export_pools()"""

    # glob all the nd2 files in directory
    fps_nd2 = natsorted(data_dir.glob("*.nd2"))
    if not fps_nd2:
        raise ValueError(f"No .nd2 files found in {data_dir}")

    # loop through nd2 files for processing
    for fp in tqdm(fps_nd2):
        # find and extract pools
        try:
            finder = PoolFinder(
                filepath=fp,
                pool_radius_um=pool_radius_um,
                pool_spacing_um=pool_spacing_um
            )

        # skip over failures caused by
        # > corrupt nd2 files
        # > when extrapolation fails due to < 3 detected pools
        except ValueError as err:
            msg = f"Processing for {finder.filepath} failed:"
            print(msg, err)
            continue

        # extract pools and save output
        finder.extract_pools()
        finder.make_debug_sketch()
        finder.segment_pools()
        finder.export_pools()


if __name__ == "__main__":
    main()
