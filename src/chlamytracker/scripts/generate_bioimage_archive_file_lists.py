import json
from pathlib import Path

import click
import pandas as pd
from swimtracker import cli_options
from natsort import natsorted

DATA_DIRECTORY = Path(__file__).parents[3] / "data"
SAMPLE_PREP_PARAMETERS_JSON = DATA_DIRECTORY / "experimental_parameters.json"

COLUMN_NAMES_SAMPLE_PREP_PARAMETERS = [
    "vessel_type",
    "position_in_tube",
    "time_in_water",
]
COLUMN_NAMES = ["Files", "file_content"] + COLUMN_NAMES_SAMPLE_PREP_PARAMETERS

FILE_CONTENT_BLURBS = {
    ".nd2": "raw brightfield time-lapse microscopy data",
    ".csv": "coordinates from cell tracking",
    ".tiff": "segmented time lapse",
    ".mp4": "movie of tracked cells with animated trajectories",
    ".jpg": "image of detected pools",
    ".txt": "detected pool positions",
}


def generate_AMID04_dataframe(filepath):
    """Generate the file list for AMID-04"""
    sample_prep_parameters = json.loads(SAMPLE_PREP_PARAMETERS_JSON.read_text())
    sample_prep_parameters = sample_prep_parameters["AMID-04"]

    # glob all raw + processed data files
    filepaths = (
        natsorted(filepath.glob("*/*.nd2"))
        + natsorted(filepath.glob("*/processed/*.mp4"))
        + natsorted(filepath.glob("*/processed/*/*"))
    )

    file_list_data = []
    for path in filepaths:
        # extract `slide_id` from absolute path
        slide_id = path.parts[6]

        row = {
            "Files": Path(*path.parts[5:]),
            "file_content": FILE_CONTENT_BLURBS[path.suffix],
            "vessel_type": sample_prep_parameters[slide_id]["vessel_type"],
            "position_in_tube": sample_prep_parameters[slide_id]["position_in_tube"],
            "time_in_water": sample_prep_parameters[slide_id]["time_in_water"],
            "slide_id": slide_id,
        }
        file_list_data.append(row)

    columns = COLUMN_NAMES + ["slide_id"]
    dataframe = pd.DataFrame(file_list_data, columns=columns)
    return dataframe


def generate_AMID05_dataframe(filepath):
    """Generate the file list for AMID-05."""
    sample_prep_parameters = json.loads(SAMPLE_PREP_PARAMETERS_JSON.read_text())
    sample_prep_parameters = sample_prep_parameters["AMID-05"]

    # glob all raw + processed data files
    filepaths = natsorted(filepath.glob("*.nd2")) + natsorted(filepath.glob("processed/*"))

    file_list_data = []
    for path in filepaths:
        # extract `well_id` from filename :: `Well{well_id}_Point{well_id}_{sequence}_...`
        well_id = path.name[4:7]

        row = {
            "Files": Path(*path.parts[5:]),
            "file_content": FILE_CONTENT_BLURBS[path.suffix],
            "vessel_type": sample_prep_parameters[well_id]["vessel_type"],
            "position_in_tube": sample_prep_parameters[well_id]["position_in_tube"],
            "time_in_water": sample_prep_parameters[well_id]["time_in_water"],
            "well_id": well_id,
        }
        file_list_data.append(row)

    columns = COLUMN_NAMES + ["well_id"]
    dataframe = pd.DataFrame(file_list_data, columns=columns)
    return dataframe


@click.command()
@cli_options.input_directory_argument
def main(input_directory):
    """Create a file list to accompany data upload to BioImage Archive.

    BioImage Archive requires a File List [1] to accompany each study component [2] you upload.
    According to their website,
    | A File List is used to describe all the files that you wish to include in your submission,
    | both image files and other supporting files e.g., analysis results. It contains file level
    | metadata.

    The full list of rules regarding the File List is available via their website, but the ones
    that are relevant for programmatically generating the File List are
    | * File lists are File lists are tabular data, either in tsv or Excel (.xlsx) format.
    | * The first column of the header has to be the word “Files”.
    | * File path separator must be forward slash “/”.
    | * Allowed characters :: a-z A-Z 0-9 !-_.*'()

    References
    ----------
    [1] https://www.ebi.ac.uk/bioimage-archive/help-file-list/
    [2] https://www.ebi.ac.uk/bioimage-archive/rembi-help-examples/
    """
    # assume subdirectories of input directory correspond to study components
    study_component_directories = natsorted(
        [directory for directory in input_directory.glob("*") if directory.is_dir()]
    )

    for directory in study_component_directories:
        study_component_name = directory.name

        if "AMID-04" in study_component_name:
            dataframe = generate_AMID04_dataframe(directory)
        elif "AMID-05" in study_component_name:
            dataframe = generate_AMID05_dataframe(directory)
        else:
            msg = "Unknown study component '{study_component_name}'."
            raise ValueError(msg)

        tsv_filename = DATA_DIRECTORY / f"{study_component_name}_file-list.tsv"
        dataframe.to_csv(tsv_filename, sep="\t", index=False)


if __name__ == "__main__":
    main()
