from pathlib import Path

import click

data_dir_option = click.option(
    "--dir-data", type=Path, help="Path to the data directory"
)
data_file_option = click.option(
    "--filename", type=Path, help="Path to the file location"
)
