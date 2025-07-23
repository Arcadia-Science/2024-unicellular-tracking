from pathlib import Path

import click

input_directory_argument = click.argument(
    "input_directory",
    type=Path,
)

output_directory_option = click.option(
    "--output-directory",
    "output_directory",
    type=Path,
    default=None,
    show_default=False,
    help="Path to directory in which to save output.",
)

glob_option = click.option(
    "--glob",
    "glob_str",
    type=str,
    default="*.nd2",
    show_default=True,
    help="A glob like string that may match one or multiple filenames.",
)

verbose_option = click.option(
    "-v",
    "--verbose",
    "verbose",
    is_flag=True,
    help="Whether to output lots of neat information.",
)

num_workers_option = click.option(
    "--num-workers",
    "num_workers",
    type=int,
    default=6,
    show_default=True,
    help="Number of workers to use for parallel processing.",
)

use_dask_option = click.option(
    "--use-dask",
    "use_dask",
    is_flag=True,
    help="Whether to use `dask` for parallelization.",
)

btrack_config_file_option = click.option(
    "--config-file",
    "btrack_config_file",
    type=Path,
    default=None,
    show_default=False,
    help="Path to btrack configuration file.",
)

framerate_option = click.option(
    "--framerate",
    "framerate",
    type=int,
    default=30,
    show_default=True,
    help="Framerate of output Napari animation.",
)

min_cell_diameter_um_option = click.option(
    "--min-cell-diameter",
    "min_cell_diameter_um",
    type=float,
    default=6,
    show_default=True,
    help="Diameter [µm] of smallest desired organism to be segmented.",
)

pixelsize_um_option = click.option(
    "--pixelsize_um",
    "pixelsize_um",
    type=float,
    default=None,
    help="Pixelsize [µm] of timelapse. Required for processing TIFF files.",
)

frametime_s_option = click.option(
    "--frametime_s",
    "frametime_s",
    type=float,
    default=None,
    help="The time [s] between frames of the timelapse. Required for processing TIFF files.",
)
