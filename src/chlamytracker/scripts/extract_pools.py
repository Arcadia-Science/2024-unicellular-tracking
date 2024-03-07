import click
from chlamytracker import cli_options
from chlamytracker.pool_finder import PoolFinder
from natsort import natsorted
from tqdm import tqdm


@cli_options.data_dir_option
@click.command()
def main(dir_data):
    """Wrapper for PoolFinder.extract_pools() and PoolFinder.export_pools()"""

    # glob all the nd2 files in directory
    fps_nd2 = natsorted(dir_data.glob("*.nd2"))
    if not fps_nd2:
        raise ValueError(f"No .nd2 files found in {dir_data}")

    # loop through nd2 files for processing
    for fp in tqdm(fps_nd2[:7]):
        # find and extract pools
        try:
            finder = PoolFinder(filepath=fp, pool_radius_um=50, pool_spacing_um=200)

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
        finder.preprocess_pools()
        finder.export_pools()


if __name__ == "__main__":
    main()
