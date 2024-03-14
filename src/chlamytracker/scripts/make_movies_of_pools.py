import re

import click
import napari
import nd2
import numpy as np
import pandas as pd
import skimage as ski
from chlamytracker import cli_options
from napari_animation import Animation
from natsort import natsorted
from tqdm import tqdm


def make_napari_animation_for_timelapse(
    filename,
    nd2_file,
    tiff_files,
    csv_files,
    txt_file,
    framerate=20,
):
    """Function for ...

    Parameters
    ----------
    filename : Path
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
    """
    # load timelapse
    timelapse = nd2.ND2File(nd2_file)
    num_frames = timelapse.sizes["T"]

    # load poolmap (format: ix iy cx cy status)
    columns = ["ix", "iy", "cx", "cy", "status"]
    poolmap = pd.read_csv(txt_file, sep="\\s+", header=None, names=columns)

    # create napari viewer
    viewer = napari.Viewer(show=True)
    viewer.add_image(timelapse.asarray()[:, np.newaxis, :, :], name=nd2_file.stem)
    timelapse.close()

    # loop through data for each pool
    for tiff_file, csv_file in zip(tiff_files, csv_files, strict=False):
        # get pool index
        ix, iy = (int(i) for i in re.findall("\\d+", tiff_file.stem))

        # load segmentation
        cells = ski.io.imread(tiff_file)
        # cells_labelled = ski.measure.label(cells)

        # load tracks (format: ID t x y z)
        df = pd.read_csv(csv_file, sep="\\s+", header=None, skiprows=1)
        # napari format: ID,T,(Z),Y,X
        tracks = df[[0, 1, 4, 3, 2]].values

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

    current_step = viewer.dims.current_step
    viewer.dims.current_step = (0, *current_step[1:])
    animation.capture_keyframe()
    viewer.dims.current_step = (num_frames, *current_step[1:])
    animation.capture_keyframe(steps=120)

    animation.animate(filename, fps=framerate, canvas_only=True)
    viewer.close()

    # napari.run()


@cli_options.data_dir_option
@click.command()
def main(data_dir):
    """Script for batch processing ...

    The following data files are needed to make a movie for each .nd2 timelapse
        - {timelapse}.nd2
        - pool_{ix}_{iy}.tiff
        - pool_{ix}_{iy}_tracks.csv
        - poolmap.txt
    """

    # glob all .nd2 files in directory
    nd2_files = natsorted(data_dir.glob("*.nd2"))
    if not nd2_files:
        raise ValueError(f"No .nd2 files found in {data_dir}")

    # loop through .nd2 files
    for nd2_file in tqdm(nd2_files):
        # collect data files
        source_dir = nd2_file.parent / "processed" / nd2_file.stem
        tiff_files = natsorted(source_dir.glob("pool*.tiff"))
        csv_files = natsorted(source_dir.glob("pool*_tracks.csv"))
        txt_file = source_dir / "poolmap.txt"

        # make de movie
        filename = nd2_file.parent / f"{nd2_file.stem}.mp4"
        make_napari_animation_for_timelapse(
            filename,
            nd2_file,
            tiff_files,
            csv_files,
            txt_file,
        )


if __name__ == "__main__":
    main()
