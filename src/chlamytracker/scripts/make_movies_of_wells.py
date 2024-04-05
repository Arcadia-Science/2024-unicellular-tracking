import logging

import click
import napari
import nd2
import numpy as np
import pandas as pd
from chlamytracker import cli_api
from chlamytracker.utils import crop_movie_to_content
from napari_animation import Animation
from natsort import natsorted
from tqdm import tqdm

logger = logging.getLogger(__name__)


def make_napari_animation_for_timelapse(
    filename,
    nd2_file,
    tiff_file,
    csv_file,
    framerate=20,
):
    """Renders a napari animation of tracked cells and overlays it onto the
    raw timelapse data.

    Parameters
    ----------
    filename : Path
        Output filename for animation.
    nd2_file : Path
        Input timelapse microscopy data of tiny organisms swimming around in a well.
    tiff_file : Path
        Tiff file of segmented cells corresponding to the nd2 file.
    csv_file : Path
        Csv file of motility data corresponding to the nd2 file.
    """
    # load timelapse and metadata
    timelapse = nd2.imread(nd2_file)
    with nd2.ND2File(nd2_file) as nd2f:
        timelapse = nd2f.asarray()
        num_frames = nd2f.sizes["T"]

    # create napari viewer
    viewer = napari.Viewer(show=False)
    viewer.add_image(timelapse[:, np.newaxis, :, :], name=nd2_file.stem)

    # load tracks (format: ID t x y z)
    df = pd.read_csv(csv_file, sep="\\s+", header=None, skiprows=1)
    # napari format: ID,T,(Z),Y,X
    tracks = df[[0, 1, 4, 3, 2]].values
    # add tracks to napari viewer
    viewer.add_tracks(tracks, name=csv_file.stem)

    # make movie with napari
    animation = Animation(viewer)

    # set first and last frames as key frames
    current_step = viewer.dims.current_step
    viewer.dims.current_step = (0, *current_step[1:])
    animation.capture_keyframe()
    viewer.dims.current_step = (num_frames, *current_step[1:])
    animation.capture_keyframe(steps=120)

    animation.animate(filename, fps=framerate, canvas_only=True)
    viewer.close()


@cli_api.glob_option
@cli_api.output_directory_option
@cli_api.input_directory_argument
@click.command()
def main(
    input_directory,
    output_directory,
    glob_str,
    framerate=20,
):
    """Script for batch processing napari animations of tracked cells.

    The following data files are needed to make a movie for each nd2 file
      - {timelapse}.nd2
      - {timelapse}_segmented.tiff
      - {timelapse}_tracks.csv

    Searches {input_directory} for __ and {output_directory} for the
    corresponding tiff and csv files.
    """

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

        # find tiff and csv files
        tiff_file = output_directory / f"{nd2_file.stem}_segmented.tiff"
        csv_file = output_directory / f"{nd2_file.stem}_tracks.csv"

        # handle case for no tiff or csv file found
        if not (tiff_file.exists() and csv_file.exists()):
            logger.warning(f"No tiff or csv file corresponding to {nd2_file} found.")
            continue

        # create napari animation
        mp4_file = output_directory / f"{nd2_file.stem}_animation.mp4"
        make_napari_animation_for_timelapse(
            mp4_file,
            nd2_file,
            tiff_file,
            csv_file,
            framerate,
        )

        # crop borders
        crop_movie_to_content(mp4_file, framerate)


if __name__ == "__main__":
    main()
