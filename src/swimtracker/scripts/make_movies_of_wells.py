from __future__ import annotations
import logging
from pathlib import Path

import click
import napari
import numpy as np
from napari_animation import Animation
from natsort import natsorted
from swimtracker import cli_options
from swimtracker.tracking_metrics import TrajectoryCSVParser
from swimtracker.utils import configure_logger, crop_movie_to_content
from tqdm import tqdm

from .track_cells import create_timelapse

logger = logging.getLogger(__name__)


def make_napari_animation_for_timelapse(
    mp4_file: Path,
    input_file: Path,
    csv_file: Path,
    pixelsize_um: float | None = None,
    frametime_s: float | None = None,
    framerate: int = 20,
) -> None:
    """Render a napari animation of tracked cells overlaid on timelapse data.

    Creates smooth transitions between keyframes of the napari UI canvas,
    interpolating between the first and last frame of the timelapse.

    Args:
        mp4_file: Output filename for animation.
        input_file: Input timelapse file (ND2 or TIFF).
        csv_file: CSV file of motility data corresponding to the input file.
        pixelsize_um: Pixel size in micrometers (required for TIFF files).
        frametime_s: Frame time in seconds (required for TIFF files).
        framerate: Animation framerate in frames per second.

    Raises:
        FileNotFoundError: If input files don't exist.
        ValueError: If file formats are unsupported or parameters are missing.

    References:
        [1] https://en.wikipedia.org/wiki/Key_frame
        [2] https://napari.org/napari-animation/index.html
    """
    # load timelapse and metadata
    logger.info(f"Loading timelapse file {input_file}...")
    timelapse_obj = create_timelapse(input_file, pixelsize_um, frametime_s)
    timelapse = timelapse_obj.raw_data
    num_frames = timelapse_obj.num_frames

    # create napari viewer
    viewer = napari.Viewer(show=True)
    # Handle different dimensionalities - add singleton dimension if needed
    if timelapse.ndim == 3:  # TYX format
        viewer.add_image(timelapse[:, np.newaxis, :, :], name=input_file.stem)
    else:  # Assume already in correct format
        viewer.add_image(timelapse, name=input_file.stem)

    # resize napari window
    width_px = 1400
    height_px = 1200
    viewer.window.resize(width_px, height_px)

    # Load and validate tracking data
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

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
@cli_options.pixelsize_um_option
@cli_options.frametime_s_option
@cli_options.verbose_option
def main(
    input_directory: Path,
    output_directory: Path | None,
    framerate: int,
    glob_str: str,
    pixelsize_um: float,
    frametime_s: float,
    verbose: bool,
) -> None:
    """Script for batch processing napari animations of tracked cells in multi-well plates.

    Creates MP4 animations showing cell tracks overlaid on the original timelapse data.

    Required files for each timelapse:
        - {timelapse}.nd2 or {timelapse}.tiff (raw timelapse data)
        - {timelapse}_tracks.csv (tracking results)

    Args:
        input_directory: Directory containing timelapse files.
        output_directory: Directory to save animations (defaults to input_directory/processed).
        framerate: Animation framerate in FPS.
        glob_str: Glob pattern to match input files.
        pixelsize_um: Pixel size in micrometers (required for TIFF files).
        frametime_s: Frame time in seconds (required for TIFF files).
        verbose: Enable verbose logging.
    """
    if verbose:
        configure_logger()

    # glob all timelapse files in directory
    input_files = natsorted(input_directory.glob(glob_str))
    if not input_files:
        logger.error(f"No files found matching '{glob_str}' in {input_directory}.")
        return

    # ensure output directory exists
    if output_directory is None:
        output_directory = input_directory / "processed"
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create output directory {output_directory}: {e}")
        return

    # loop through timelapse files
    for input_file in tqdm(input_files):
        try:
            # find csv file
            csv_file = output_directory / f"{input_file.stem}_tracks.csv"

            # handle case for no csv file found
            if not csv_file.exists():
                logger.warning(f"No CSV file corresponding to {input_file} found.")
                continue

            # create napari animation
            mp4_file = output_directory / f"{input_file.stem}_animation.mp4"
            make_napari_animation_for_timelapse(
                mp4_file,
                input_file,
                csv_file,
                pixelsize_um,
                frametime_s,
                framerate,
            )

            # crop borders
            crop_movie_to_content(mp4_file, framerate)

        except (ValueError, FileNotFoundError, OSError) as err:
            logger.warning(f"Animation creation for {input_file} failed: {err}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
