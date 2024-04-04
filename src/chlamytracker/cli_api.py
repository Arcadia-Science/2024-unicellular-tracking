from pathlib import Path

import click

input_directory_argument = click.argument("input_directory", type=Path)

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

log_level_option = click.option(
    "--log-level", "log_level", type=str, default="INFO", show_default=True, help="Log level."
)

use_dask_option = click.option(
    "--use-dask",
    "use_dask",
    type=bool,
    default=False,
    show_default=True,
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

min_cell_diameter_um_option = click.option(
    "--min-cell-diameter",
    "min_cell_diameter_um",
    type=float,
    default=6,
    show_default=True,
    help="Diameter [um] of smallest desired organism to be segmented.",
)

pool_radius_um_option = click.option(
    "--pool-radius",
    "pool_radius_um",
    type=float,
    default=50,
    show_default=True,
    help="Radius [um] of agar microchamber pool.",
)
pool_spacing_um_option = click.option(
    "--pool-spacing",
    "pool_spacing_um",
    type=float,
    default=200,
    show_default=True,
    help="Distance [um] between adjacent microchamber pools (measured from center to center).",
)
