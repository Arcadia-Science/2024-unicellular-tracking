import logging
import re

import click
import napari
import nd2
import numpy as np
import pandas as pd
import skimage as ski
from napari_animation import Animation
from natsort import natsorted
from swimtracker import cli_options
from swimtracker.tracking_metrics import TrajectoryCSVParser
from swimtracker.utils import configure_logger, crop_movie_to_content
from tqdm import tqdm

logger = logging.getLogger(__name__)


def make_napari_animation_for_timelapse(
    mp4_file,
    nd2_file,
    tiff_csv_file_pairs,
    txt_file,
    framerate=20,
):
    """Renders a napari animation of tracked cells and overlays it onto the
    raw timelapse data.

    See `make_movies_of_wells.make_napari_animation_for_timelapse()` for details
    on how the animation is rendered.

    Parameters
    ----------
    mp4_file : Path
        Output filename for animation.
    nd2_file : Path
        Input timelapse microscopy data of tiny organisms swimming around in pools.
    tiff_csv_file_pairs : list
        List of tiff files of segmented pools and corresponding csv files of
        motility data from each pool.
    txt_file : Path
        Text file (`poolmap.txt`) that contains the indices and coordinates of
        each pool detected from the timelapse.
    framerate : int (optional)
        Frame rate for the animation.
    """
    # load timelapse and metadata
    with nd2.ND2File(nd2_file) as nd2f:
        timelapse = nd2f.asarray()
        num_frames = nd2f.sizes["T"]

    # load poolmap (format: ix iy cx cy status)
    columns = ["ix", "iy", "cx", "cy", "status"]
    poolmap = pd.read_csv(txt_file, sep="\\s+", header=None, names=columns)

    # create napari viewer
    viewer = napari.Viewer(show=True)
    viewer.add_image(timelapse[:, np.newaxis, :, :], name=nd2_file.stem)

    # resize napari window
    width_px = 1400
    height_px = 1200
    viewer.window.resize(width_px, height_px)

    # loop through data for each pool
    for tiff_file, csv_file in tiff_csv_file_pairs:
        # get pool index
        ix, iy = (int(i) for i in re.findall("\\d+", tiff_file.stem))

        # load segmentation
        cells = ski.io.imread(tiff_file)
        # cells_labelled = ski.measure.label(cells)

        df = TrajectoryCSVParser(csv_file).dataframe
        # napari format: ID,T,(Z),Y,X
        tracks = df[["ID", "t", "z", "y", "x"]].values

        # shift tracks to center of each pool
        pool_coords = poolmap.loc[(poolmap["ix"] == ix) & (poolmap["iy"] == iy)].iloc[0]
        cx = pool_coords["cx"]
        cy = pool_coords["cy"]
        tracks[:, 4] += cx - cells.shape[2] // 2
        tracks[:, 3] += cy - cells.shape[1] // 2

        # add data to napari viewer
        # viewer.add_labels(cells_labelled[:, np.newaxis, :, :])
        viewer.add_tracks(tracks, name=csv_file.stem)

    # make movie with napari
    animation = Animation(viewer)

    # set first and last frames as key frames
    current_step = viewer.dims.current_step
    viewer.dims.current_step = (0, *current_step[1:])
    animation.capture_keyframe()
    viewer.dims.current_step = (num_frames, *current_step[1:])
    animation.capture_keyframe(steps=num_frames)

    animation.animate(mp4_file, fps=framerate, canvas_only=True)
    viewer.close()


@click.command()
@cli_options.input_directory_argument
@cli_options.output_directory_option
@cli_options.framerate_option
@cli_options.glob_option
@cli_options.verbose_option
def main(
    input_directory,
    output_directory,
    framerate,
    glob_str,
    verbose,
):
    """Script for batch processing napari animations of tracked cells in 384 or
    1536 well plates.

    The following data files are needed to make a movie for each nd2 file
      - {timelapse}.nd2
      - {timelapse}/pool_{x}_{y}_segmented.tiff(s)
      - {timelapse}/pool_{x}_{y}_tracks.csv(s)

    Searches {input_directory} for nd2 files of the raw timelapse data
    and {output_directory} for corresponding tiff and csv files.
    """
    if verbose:
        configure_logger()

    # glob all .nd2 files in directory
    nd2_files = natsorted(input_directory.glob(glob_str))
    if not nd2_files:
        raise ValueError(f"No nd2 files found in {input_directory}.")

    # ensure output directory exists
    if output_directory is None:
        output_directory = input_directory / "processed"
    if not output_directory.exists():
        msg = f"Output directory: {output_directory} does not exist."
        raise FileNotFoundError(msg)

    # loop through nd2 files
    for nd2_file in tqdm(nd2_files):
        # find corresponding tiff files and `poolmap.txt`
        output_subdirectory = output_directory / nd2_file.stem
        tiff_files = natsorted(output_subdirectory.glob("*_segmented.tiff"))
        txt_file = output_subdirectory / "poolmap.txt"

        # missing tiff files --> skip
        if not tiff_files:
            logger.warning(
                f"Processing for {nd2_file.name} failed: no tiff files of segmented cells found."
            )
            continue
        # missing `poolmap.txt` --> skip
        if not txt_file.exists():
            logger.warning(f"Processing for {nd2_file.name} failed: {txt_file} not found.")
            continue

        # find csv files corresponding to each tiff
        tiff_csv_file_pairs = []
        for tiff_file in tiff_files:
            # construct the expected csv file (which may or may not exist)
            ix, iy = (int(i) for i in re.findall("\\d+", tiff_file.stem))
            csv_file = tiff_file.parent / f"pool_{ix}_{iy}_tracks.csv"
            if csv_file.exists():
                tiff_csv_file_pairs.append((tiff_file, csv_file))

        # missing corresponding csv files --> skip
        if not tiff_csv_file_pairs:
            logger.warning(
                f"Processing for {nd2_file.name} failed: no corresponding csv "
                "files of cell trajectories found."
            )
            continue

        # create napari animation
        mp4_file = output_directory / f"{nd2_file.stem}_animation.mp4"
        make_napari_animation_for_timelapse(
            mp4_file,
            nd2_file,
            tiff_csv_file_pairs,
            txt_file,
        )

        # crop borders
        crop_movie_to_content(mp4_file, framerate)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
