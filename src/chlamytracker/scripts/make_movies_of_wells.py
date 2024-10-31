import logging

import click
import napari
import nd2
import numpy as np
from swimtracker import cli_options
from swimtracker.tracking_metrics import TrajectoryCSVParser
from swimtracker.utils import configure_logger, crop_movie_to_content
from napari_animation import Animation
from natsort import natsorted
from tqdm import tqdm

logger = logging.getLogger(__name__)


def make_napari_animation_for_timelapse(
    mp4_file,
    nd2_file,
    csv_file,
    framerate=20,
):
    """Renders a napari animation of tracked cells and overlays it onto the
    raw timelapse data.

    Animations are rendered by making smooth transitions between key frames [1]
    of the napari UI canvas. While more elaborate transitions are possible via
    the napari-animation API [2] (think Ken Burns style documentaries), the
    animations created here simply take the first and last frame of the
    timelapse as key frames and interpolate between them at the specified
    `framerate`.

    Parameters
    ----------
    mp4_file : Path
        Output filename for animation.
    nd2_file : Path
        Input timelapse microscopy data of tiny organisms swimming around in a well.
    csv_file : Path
        Csv file of motility data corresponding to the nd2 file.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Key_frame
    [2] https://napari.org/napari-animation/index.html
    """
    # load timelapse and metadata
    logger.info(f"Loading nd2 file {nd2_file}...")
    with nd2.ND2File(nd2_file) as nd2f:
        timelapse = nd2f.asarray()
        num_frames = nd2f.sizes["T"]

    # create napari viewer
    viewer = napari.Viewer(show=True)
    viewer.add_image(timelapse[:, np.newaxis, :, :], name=nd2_file.stem)

    # resize napari window
    width_px = 1400
    height_px = 1200
    viewer.window.resize(width_px, height_px)

    df = TrajectoryCSVParser(csv_file).dataframe
    # napari format: ID,T,(Z),Y,X
    tracks = df[["ID", "t", "z", "y", "x"]].values
    # add tracks to napari viewer
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
      - {timelapse}_segmented.tiff
      - {timelapse}_tracks.csv

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
        # find tiff and csv files
        csv_file = output_directory / f"{nd2_file.stem}_tracks.csv"

        # handle case for no tiff or csv file found
        if not csv_file.exists():
            logger.warning(f"No csv file corresponding to {nd2_file} found.")
            continue

        # create napari animation
        mp4_file = output_directory / f"{nd2_file.stem}_animation.mp4"
        make_napari_animation_for_timelapse(
            mp4_file,
            nd2_file,
            csv_file,
            framerate,
        )

        # crop borders
        crop_movie_to_content(mp4_file, framerate)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
