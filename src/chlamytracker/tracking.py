from pathlib import Path

import btrack

CONFIG_FILE = Path(__file__).parents[2] / "btrack_config/cell_config.json"


class Tracker:
    """Class for tracking cell motility within agar microchamber pools.

    Parameters
    ----------
    segmentation_data : (T, [Z], Y, X) uint8 numpy array
        Segmented image data to input for cell tracking. Passed without
        modification to `btrack.utils.segmentation_to_objects` [1]. Thus, other
        forms of input are also accepted.
    config_file : str or `pathlib.Path`
        Filepath to configuration file for btrack parameters [2].
    additional_properties : str or list-like
        Additional object properties to record during tracking. Additional
        properties are passed to `skimage.measure.regionprops` [3] via
        `btrack.utils.segmentation_to_objects`.

    References
    ----------
    [1] https://btrack.readthedocs.io/en/latest/api/btrack.io.segmentation_to_objects.html#btrack.io.segmentation_to_objects
    [2] https://btrack.readthedocs.io/en/latest/user_guide/configuration.html
    [3] https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
    """

    def __init__(self, segmentation_data, additional_properties=None, config_file=None):
        self.segmentation_data = segmentation_data
        self.properties = self.set_properties(additional_properties)
        self.config_file = CONFIG_FILE if config_file is None else config_file

        self.trackable_objects = btrack.io.segmentation_to_objects(
            segmentation_data, properties=self.properties
        )

    def set_properties(self, additional_properties):
        """Handles additional object properties for tracking."""

        base_properties = (
            "area",
            "eccentricity",
            "major_axis_length",
            "minor_axis_length",
            "orientation",
            "perimeter",
            "solidity",
        )

        if additional_properties is None:
            properties = base_properties
        elif isinstance(additional_properties, str):
            properties = base_properties + (additional_properties,)
        elif isinstance(additional_properties, list) or isinstance(additional_properties, tuple):
            properties = base_properties + tuple(additional_properties)
        else:
            msg = (
                "Expected str or list (or tuple) of str for `additional_properties`, "
                f"but received {type(additional_properties)}."
            )
            raise TypeError(msg)

        return properties

    def track_cells(self):
        """Track cells.

        Wrapper for `btrack.BayesianTracker().track()`.
        """
        self.tracker = btrack.BayesianTracker(verbose=True)
        self.tracker.configure(self.config_file)
        self.tracker.append(self.trackable_objects)
        self.tracker.track()
