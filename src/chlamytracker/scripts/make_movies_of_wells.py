import logging

import click
import napari
import nd2
import numpy as np
import pandas as pd
from chlamytracker import cli_options
from chlamytracker.utils import crop_movie_to_content
from napari_animation import Animation
from natsort import natsorted
from tqdm import tqdm


def make_napari_animation_for_timelapse(
    filename,
    nd2_file,
    csv_file,
    framerate=20,
):
    """Function for ...

    Parameters
    ----------
    filename : Path
        Output filename for animation.
    nd2_file : Path
        Input timelapse microscopy data of tiny organisms swimming around in a well.
    csv_file : Path
        CSV file of motility data from a well.
    """
    # load timelapse and metadata
    timelapse = nd2.imread(nd2_file)
    with nd2.ND2File(nd2_file) as nd2f:
        timelapse = nd2f.asarray()
        num_frames = nd2f.sizes["T"]

    # create napari viewer
    viewer = napari.Viewer(show=True)
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


@cli_options.data_dir_option
@click.command()
def main(data_dir, framerate=20):
    """Script for batch processing napari animations of tracked cells.

    The following data files are needed to make a movie for each nd2 file
      - {timelapse}.nd2
      - {timelapse}_tracks.csv
    """

    # glob all .nd2 files in directory
    nd2_files = natsorted(data_dir.glob("*.nd2"))
    if not nd2_files:
        raise ValueError(f"No .nd2 files found in {data_dir}")

    # loop through .nd2 files
    for nd2_file in tqdm(nd2_files):
        # find csv file
        csv_file = nd2_file.parent / "processed" / f"{nd2_file.stem}_tracks.csv"

        # skip over timelapses with missing data
        if not csv_file.exists():
            msg = f"Processing for {nd2_file.name} failed: {csv_file} not found."
            logging.error(msg)
            continue

        # skip over corrupt nd2 files
        try:
            with nd2.ND2File(nd2_file) as nd2f:
                _ = nd2f.shape
        except ValueError as err:
            msg = f"Processing for {nd2_file.name} failed:"
            logging.error(msg, exc_info=err)
            continue

        # make napari animation
        mp4_file = nd2_file.parent / f"{nd2_file.stem}.mp4"
        make_napari_animation_for_timelapse(
            mp4_file,
            nd2_file,
            csv_file,
        )

        # resize movie
        crop_movie_to_content(mp4_file, framerate)


if __name__ == "__main__":
    main()
