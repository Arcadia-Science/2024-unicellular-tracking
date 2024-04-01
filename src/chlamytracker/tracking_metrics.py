import numpy as np
import pandas as pd


class TrackingDataAnalyzer:
    """Class for tracking cell motility within agar microchamber pools.

    Parameters
    ----------
    csv_file : str or `pathlib.Path`
        Path to csv file for analyzing.
    additional_properties : str or list-like
        Additional object properties to record during tracking. Additional
        properties are passed to `skimage.measure.regionprops` [3] via
        `btrack.utils.segmentation_to_objects`.
    """

    def __init__(self, csv_file, additional_properties=None):
        self.properties = self.set_properties(additional_properties)
        self.dataframe = pd.read_csv(csv_file, sep="\\s+", names=self.properties, skiprows=1)

    def set_properties(self, additional_properties):
        """Handles additional object properties for tracking."""

        default_btrack_csv_columns = [
            "ID",
            "t",
            "x",
            "y",
            "z",
            "parent",
            "root",
            "state",
            "generation",
            "dummy",
        ]

        default_base_properties = [
            "area",
            "eccentricity",
            "major_axis_length",
            "minor_axis_length",
            "orientation",
            "perimeter",
            "solidity",
        ]

        if additional_properties is None:
            properties = default_btrack_csv_columns + default_base_properties
        elif (
            isinstance(additional_properties, str)
            or isinstance(additional_properties, list)
            or isinstance(additional_properties, tuple)
        ):
            properties = (
                default_btrack_csv_columns + default_base_properties + list(additional_properties)
            )
        else:
            msg = (
                "Expected str or list (or tuple) of str for `additional_properties`, "
                f"but received {type(additional_properties)}."
            )
            raise TypeError(msg)

        return properties

    def get_num_cells(self):
        num_cells = 3

        return num_cells
