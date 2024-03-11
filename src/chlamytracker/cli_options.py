from pathlib import Path

import click

data_dir_option = click.option("--data-dir", type=Path, help="Path to the data directory")
data_file_option = click.option("--filename", type=Path, help="Path to the file location")

pool_radius_um_option = click.option(
    "--pool-radius",
    type=float,
    default=50,
    show_default=True,
    help="Radius of agar microchamber pool.",
)
pool_spacing_um_option = click.option(
    "--pool-spacing",
    type=float,
    default=200,
    show_default=True,
    help="Distance between adjacent microchamber pools (measured from center to center).",
)
