import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TrajectoryCSVParser:
    """Class for tracking cell motility within agar microchamber pools.

    Parameters
    ----------
    csv_file : str or `pathlib.Path`
        Path to csv file of motility data for analysis.
    framerate : float (optional)
        Frame rate [frames per second (fps)] of timelapse from which motility
        data originates. Equal to 1000 / exposure time when exposure time is
        measured in milliseconds.
    pixelsize : float (optional)
        Pixel size [µm / px] of timelapse from which motility data originates
        (assumes square pixels).
    min_frames : int (optional)
        Minimum number of frames required for measuring a trajectory.
    """

    def __init__(self, csv_file, framerate=None, pixelsize=None, min_frames=5):
        # This is kind of awkward but basically the tracking data is output as
        # a csv file in one of two styles. The first basically follows the btrack
        # default plus a few extra columns for the properties such as area,
        # eccentricity, etc. The second uses `Tracker.to_dataframe().to_csv()`.
        # There are a few ways to differentiate between the two styles, but the
        # main/easiest way is that only the second style is actually comma
        # delimited. For whatever reason the btrack export outputs csvs as white
        # space delimited.
        with open(csv_file) as csv:
            csv_headers = next(csv)

        if len(csv_headers.split(" ")) > 1:
            properties = self.set_properties()
            self.dataframe = pd.read_csv(csv_file, sep="\\s+", names=properties, skiprows=1)
        elif len(csv_headers.split(",")) > 1:
            self.dataframe = pd.read_csv(csv_file)
        else:
            raise ValueError(f"Unable to read csv file {csv_file}.")

        if framerate is not None:
            self.framerate = framerate
            self.frametime = 1 / framerate
        if pixelsize is not None:
            self.pixelsize = pixelsize
        self.min_frames = min_frames

    def set_properties(self):
        """Handles additional object properties for tracking."""

        # btrack.constants.DEFAULT_EXPORT_PROPERTIES
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

        properties = default_btrack_csv_columns + default_base_properties
        return properties

    def get_trajectories(self):
        """Get individual cell trajectories from motility data by ID number.

        Returns
        -------
        cell_trajectories : (N, 2) float array
            Cell trajectories as a series of (x, y) coordinates.
        """
        cell_trajectories = []
        for _cell_id, cell_data in self.dataframe.groupby("ID"):
            x = cell_data["x"].values
            y = cell_data["y"].values
            trajectory = np.array([x, y]).T
            cell_trajectories.append(trajectory)

        return cell_trajectories

    def measure_trajectories(self):
        """

        Returns
        -------
        measurement_collection : list
            List of trajectory measurements
        """
        if not hasattr(self, "framerate") or not hasattr(self, "pixelsize"):
            msg = (
                "Must initialize `TrajectoryCSVParser` with framerate and "
                "pixel size in order to make trajectory measurements."
            )
            raise AttributeError(msg)

        measurement_collection = []
        for cell_id, cell_data in self.dataframe.groupby("ID"):
            x = cell_data["x"].values
            y = cell_data["y"].values

            if x.size < self.min_frames:
                logger.debug(
                    f"Discarding cell {cell_id} with fewer than {self.min_frames} "
                    "points in its trajectory."
                )
                continue

            trajectory_analyzer = TrajectoryAnalyzer(
                x, y, seconds=self.frametime, microns=self.pixelsize
            )
            measurements = trajectory_analyzer.measurements
            measurements["cell_id"] = cell_id
            measurement_collection.append(measurements)

        return measurement_collection

    def get_trajectory_count(self):
        """Get number of cell trajectories in motility data."""
        return self.dataframe["ID"].unique().size

    def estimate_cell_count(self):
        """Estimate the number of cells in the motility data.

        In general this will be somewhat close but not equal to the trajectory
        count since new trajectories are created each time cells collide or go
        in and out of focus.
        """
        num_cells = 3

        return num_cells


class TrajectoryAnalyzer:
    """Class for analyzing cell trajectories.

    Motility measurements adapted and expanded from [1].
    +--------------------------+----------------------------------+---------+
    | Variable                 | Equation                         | Unit    |
    +--------------------------+----------------------------------+---------+
    | total_time               | t_tot = N * dt                   | s       |
    | total_distance           | d_tot = sum( d(p_i, p_i+1) )     | µm      |
    | net_distance             | d_net = d(p_0, p_N)              | µm      |
    | max_distance             | d_max = max( d(p_i, p_i+1) )     | µm      |
    | confinement_ratio        | r_con = d_net / d_tot            |         |
    | mean_curvilinear_speed   | v_avg = 1 / N * sum( v_i )       | µm / s  |
    | mean_linear_speed        | v_lin = d_net / t_tot            | µm / s  |
    | mean_angular_velocity    | v_ang =                          | rad / s |
    | linearity_of_progression | r_lin = v_lin / v_avg            |         |
    | num_direction_changes    | n_chg = sum( sign(v_i+1 - v_i) ) |         |
    +--------------------------+----------------------------------+---------+

    Parameters
    ----------
    x : 1D float array
        X coordinates with respect to time.
    y : 1D float array
        Y coordinates with respect to time.
    seconds : float (optional)
        Time increment [s] between sequential (x, y) data points.
        Frame time of timelapse from which motility data originates.
    microns : float (optional)
        Distance [µm] that a cell moves for each integer step in (x, y).
        Pixel size [µm / px] of timelapse from which motility data originates
        (assumes square pixels).

    References
    ----------
    [1] https://doi.org/10.1016/B978-0-12-391857-4.00009-4
    """

    def __init__(self, x, y, seconds=1, microns=1) -> None:
        # create time axis based on number of points in trajectory
        num_points = x.size
        self.time_seconds = np.arange(num_points) * seconds  # t [s]
        # calibrated trajectory
        self.x_position_microns = x * microns  # x(t) [µm]
        self.y_position_microns = y * microns  # y(t) [µm]
        self.xy_points_microns = np.array([self.x_position_microns, self.y_position_microns]).T

        # calculate point-to-point distances to facilitate motility measurements
        distances_L1_microns = np.diff(self.xy_points_microns, axis=0)
        distances_L2_microns = np.linalg.norm(distances_L1_microns, axis=1)

        # direction and angle-based calculations
        x_velocity = np.diff(self.x_position_microns)
        y_velocity = np.diff(self.y_position_microns)
        angle = np.arctan2(y_velocity, x_velocity)
        directional_change = np.diff(angle)
        x_direction_change = np.abs(np.diff(np.sign(x_velocity))) / 2
        y_direction_change = np.abs(np.diff(np.sign(y_velocity))) / 2

        # motility measurements
        total_time = num_points * seconds
        total_distance = distances_L2_microns.sum()
        net_distance = np.linalg.norm(self.xy_points_microns[-1] - self.xy_points_microns[0])
        max_sprint_length = distances_L2_microns.max()
        confinement_ratio = net_distance / total_distance
        mean_curvilinear_speed = (distances_L2_microns / seconds).mean()
        mean_linear_speed = net_distance / total_time
        mean_angular_velocity = (directional_change / seconds).mean()
        linearity_of_progression = mean_linear_speed / mean_curvilinear_speed
        num_direction_changes = x_direction_change.sum() + y_direction_change.sum()

        self.measurements = {
            "total_time": total_time,
            "total_distance": total_distance,
            "net_distance": net_distance,
            "max_sprint_length": max_sprint_length,
            "confinement_ratio": confinement_ratio,
            "mean_curvilinear_speed": mean_curvilinear_speed,
            "mean_linear_speed": mean_linear_speed,
            "mean_angular_velocity": mean_angular_velocity,
            "linearity_of_progression": linearity_of_progression,
            "num_direction_changes": num_direction_changes,
        }
