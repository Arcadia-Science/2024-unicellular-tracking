from pathlib import Path
from natsort import natsorted
from tqdm import tqdm
import skimage as ski

from chlamytracker.pool_finder import PoolFinder


def main(dir_data):
    """Wrapper for PoolFinder.extract_pools()"""

    # glob all the nd2 files in directory
    fps_nd2 = natsorted(dir_data.glob("*/*.nd2"))
    if not fps_nd2:
        raise ValueError(f"No .nd2 files found in {dir_data}")

    # loop through nd2 files for processing
    for fp in tqdm(fps_nd2[:7]):

        # find and extract pools
        try:
            finder = PoolFinder(fp)
            finder.extract_pools()
        # skip over failures caused by 
        # > corrupt nd2 files
        # > when extrapoolation fails due to < 3 detected pools
        except ValueError as err:
            msg = f"Processing for {finder.filepath} failed:"
            print(msg, err)
            continue

        # save output
        finder.make_debug_sketch()
        finder.export_pools()


if __name__ == "__main__":

    dirs_data = {
        "2024-01-26": Path("/Users/ryanlane/Projects/chlamy_motility/data/2024-01-26/"),
        "2024-02-21": Path("/Volumes/Microscopy/Babu_frik/RyanL/2024-02-21/"),
        "2024-02-23": Path("/Users/ryanlane/Projects/chlamy_motility/data/2024-02-23/")
    }

    main(dirs_data["2024-01-26"])
