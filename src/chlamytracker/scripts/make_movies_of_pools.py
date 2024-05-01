import logging
import re

import click
import napari
import nd2
import numpy as np
import pandas as pd
import skimage as ski
from chlamytracker import cli_api
from chlamytracker.tracking_metrics import TrajectoryCSVParser
from chlamytracker.utils import configure_logger, crop_movie_to_content
from napari_animation import Animation
from natsort import natsorted
from tqdm import tqdm

logger = logging.getLogger(__name__)


def make_napari_animation_for_timelapse(
    mp4_file,
    nd2_file,
    tiff_files,
    csv_files,
    txt_file,
    framerate=20,
):
    """Function for ...

    Parameters
    ----------
    mp4_file : Path
        Output filename for animation.
    nd2_file : Path
        Input timelapse microscopy data of tiny organisms swimming around in pools.
    tiff_files : list
        List of tiff files of segmented pools.
    csv_files : list
        List of csv files of motility data from each pool.
    txt_file : Path
        Text file (`poolmap.txt`) that contains the indices and coordinates of
        each pool detected from the timelapse.
    framerate : int (optional)
        Frame rate for the animation.
    """
    # load timelapse and metadata
    timelapse = nd2.imread(nd2_file)
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
    for tiff_file, csv_file in zip(tiff_files, csv_files, strict=False):
        # get pool index
        ix, iy = (int(i) for i in re.findall("\\d+", tiff_file.stem))
        _ix, _iy = (int(i) for i in re.findall("\\d+", csv_file.stem))
        if (ix != _ix) or (iy != _iy):
            msg = "Tiff file and csv file indices do not match."
            raise ValueError(msg)

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
@cli_api.input_directory_argument
@cli_api.output_directory_option
@cli_api.framerate_option
@cli_api.glob_option
@cli_api.verbose_option
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
        # find tiff and csv files
        output_subdirectory = output_directory / nd2_file.stem
        tiff_files = natsorted(output_subdirectory.glob("*_segmented.tiff"))
        csv_files = natsorted(output_subdirectory.glob("*_tracks.csv"))
        txt_file = output_subdirectory / "poolmap.txt"

        # align tiff files and csv files (very ugly, please help)
        #   it is possible that a timelapse was segmented but there were no
        #   cells tracked or vice versa which would lead to a mismatch of tracks
        #   in the Napari animations
        csv_indices = [csv_file.stem.split("_")[1:3] for csv_file in csv_files]
        tiff_indices = [tiff_file.stem.split("_")[1:3] for tiff_file in tiff_files]
        csv_files_aligned = natsorted(
            [csv_file for i, csv_file in enumerate(csv_files) if csv_indices[i] in tiff_indices]
        )
        tiff_files_aligned = natsorted(
            [tiff_file for i, tiff_file in enumerate(tiff_files) if tiff_indices[i] in csv_indices]
        )

        # missing `poolmap.txt` --> skip
        if not txt_file.exists():
            logger.warning(f"Processing for {nd2_file.name} failed: {txt_file} not found.")
            continue

        # missing tiffs or csvs --> skip
        if not tiff_files or not csv_files or not txt_file.exists():
            logger.warning(f"Processing for {nd2_file.name} failed: missing tiff or csv files.")
            continue

        # create napari animation
        mp4_file = output_directory / f"{nd2_file.stem}_animation.mp4"
        make_napari_animation_for_timelapse(
            mp4_file,
            nd2_file,
            tiff_files_aligned,
            csv_files_aligned,
            txt_file,
        )

        # crop borders
        crop_movie_to_content(mp4_file, framerate)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
