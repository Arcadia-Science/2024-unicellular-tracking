from pathlib import Path

import click

data_dir_option = click.option(
    "--data-dir", "data_dir", type=Path, help="Path to the data directory."
)
data_file_option = click.option(
    "--filename", "filepath", type=Path, help="Path to the file location."
)

log_level_option = click.option(
    "--log-level",
    "log_level",
    type=str,
    default="INFO",
    show_default=True,
    help="Log level."
)

btrack_config_file_option = click.option(
    "--config-file",
    "btrack_config_file",
    type=Path,
    default="../../../btrack_config/cell_config.json",
    show_default=True,
    help="Path to btrack configuration file.",
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
