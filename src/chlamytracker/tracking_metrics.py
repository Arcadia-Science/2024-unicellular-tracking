import logging
from math import pi

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TrajectoryCSVParser:
    """Class for processing motility data from a csv file.

    Parameters
    ----------
    csv_file : Path | str
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

        # check if expected column names are white space delimited
        expected_columns = ["ID", "t", "x", "y"]
        if all(column in csv_headers.split(" ") for column in expected_columns):
            properties = self.get_btrack_output_column_names()
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

    def get_btrack_output_column_names(self):
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
        """Get individual cell trajectories from motility data.

        Returns
        -------
        cell_trajectories : list
            List of cell trajectories in the form of (x, y) coordinates stored
            as (N, 2) numpy float arrays.
        """
        cell_trajectories = []
        for _cell_id, cell_data in self.dataframe.groupby("ID"):
            x = cell_data["x"].values
            y = cell_data["y"].values
            trajectory = np.array([x, y]).T
            cell_trajectories.append(trajectory)

        return cell_trajectories

    def get_scaled_trajectories(self):
        """Get individual cell trajectories from motility data.

        Returns
        -------
        cell_trajectories : list
            List of cell trajectories in the form of (x, y) coordinates stored
            as (N, 2) numpy float arrays.
        """
        if not hasattr(self, "framerate") or not hasattr(self, "pixelsize"):
            msg = (
                "Must initialize `TrajectoryCSVParser` with framerate and "
                "pixel size in order to get scaled trajectories."
            )
            raise AttributeError(msg)

        cell_trajectories = []
        for _cell_id, cell_data in self.dataframe.groupby("ID"):
            x = cell_data["x"].values
            y = cell_data["y"].values
            trajectory_analyzer = TrajectoryAnalyzer(
                x, y, time_increment_s=self.frametime, scale_um=self.pixelsize
            )
            trajectory = trajectory_analyzer.xy_positions_um
            cell_trajectories.append(trajectory)

        return cell_trajectories

    def compute_summary_statistics(self):
        """Compute motility summary statistics for each trajectory in the motility data.

        The set of summary statistics and how they are calculated are tabulated
        in `TrajectoryAnalyzer`.

        Returns
        -------
        measurement_collection : list
            List of motility summary statistics.

        Examples
        --------
        >>> from chlamytracker.tracking_metrics import TrajectoryCSVParser
        >>> csv_file = "WellJ07_tracks.csv"
        >>> motility_data = TrajectoryCSVParser(csv_file, framerate=30, pixelsize=0.5)
        >>> motility_data.compute_summary_statistics()
        >>> columns_subset = [
                "cell_id",
                "total_distance",
                "confinement_ratio",
                "max_sprint_length",
            ]
        >>> pd.DataFrame(summary_statistics)[columns_subset].head()

        | cell_id | total_distance | confinement_ratio | max_sprint_length |
        |--------:|---------------:|------------------:|------------------:|
        |       1 |        35.5695 |          0.997105 |          2.42239  |
        |       3 |        45.0043 |          0.896259 |          1.36801  |
        |       4 |        68.2631 |          0.959883 |          0.767712 |
        |       5 |       323.361  |          0.687338 |          2.23714  |
        |       7 |       180.866  |          0.812985 |          1.13697  |
        """
        if not hasattr(self, "framerate") or not hasattr(self, "pixelsize"):
            msg = (
                "Must initialize `TrajectoryCSVParser` with framerate and "
                "pixel size in order to compute summary statistics."
            )
            raise AttributeError(msg)

        summary_statistics = []
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
                x, y, time_increment_s=self.frametime, scale_um=self.pixelsize
            )
            motility_metrics = trajectory_analyzer.compute_motility_metrics()
            motility_metrics["cell_id"] = cell_id
            summary_statistics.append(motility_metrics)

        return summary_statistics

    def get_trajectory_count(self):
        """Get number of cell trajectories in motility data."""
        return self.dataframe["ID"].unique().size

    def estimate_cell_count(self):
        """Estimate the number of cells in the motility data.

        Estimate is based on the average number of trajectories in each frame.
        In general this will not be equal to the trajectory count since new
        trajectories are created each time cells collide or come into focus.
        """
        cell_counts_per_frame = []
        # count the number of trajectories within each frame
        for _time, frame in self.dataframe.groupby("t"):
            cell_count = len(frame)
            cell_counts_per_frame.append(cell_count)

        # number of cells should be ~= average cell count per frame
        num_cells = np.mean(cell_counts_per_frame).round()
        return int(num_cells)


class TrajectoryAnalyzer:
    """Class for analyzing cell trajectories and computing motility metrics.

    Motility metrics adapted and expanded from [1].
    +--------------------------+----------------------------------+---------+
    | Variable                 | Equation                         | Unit    |
    +--------------------------+----------------------------------+---------+
    | total_time               | t_tot = N * dt                   | s       |
    | total_distance           | d_tot = sum( d(p_i, p_i+1) )     | µm      |
    | net_distance             | d_net = d(p_0, p_N)              | µm      |
    | max_sprint_length        | d_max = max( d(p_i, p_i+w) )     | µm      |
    | confinement_ratio        | r_con = d_net / d_tot            |         |
    +--------------------------+----------------------------------+---------+
    | mean_curvilinear_speed   | v_avg = d_tot / t_tot            | µm / s  |
    | mean_linear_speed        | v_lin = d_net / t_tot            | µm / s  |
    +--------------------------+----------------------------------+---------+
    | mean_angular_speed       | v_ang = r_tot / t_tot            | rad / s |
    | num_rotations            | n_rot = r_tot / 2π               |         |
    | num_direction_changes    | n_chg = sum( sign(v_i+1 - v_i) ) |         |
    | pivot_rate               | r_piv = n_chg / d_tot            | 1 / µm  |
    +--------------------------+----------------------------------+---------+

    Parameters
    ----------
    x : 1D float array
        X coordinates with respect to time.
    y : 1D float array
        Y coordinates with respect to time.
    time_increment_s : float (optional)
        Time increment [s] between sequential (x, y) data points.
        This should be equivalent to the frame time of timelapse from which
        motility data originates.
    scale_um : float (optional)
        Distance [µm] that a cell moves for each integer step in (x, y).
        This should be equivalent to the pixel size [µm / px] of timelapse from
        which motility data originates (assumes square pixels).
    window_size : int (optional)
        Window size for computing the metrics `max_sprint_length`,
        `num_direction_changes`, and `pivot_rate`, which are calculated on the
        basis of a rolling average (`max_sprint_length` is technically a rolling
        sum).

    References
    ----------
    [1] https://doi.org/10.1016/B978-0-12-391857-4.00009-4
    """

    def __init__(
        self,
        x_positions,
        y_positions,
        time_increment_s=1,
        scale_um=1,
        window_size=5,
    ):
        self.time_increment_s = time_increment_s
        self.scale_um = scale_um
        self.window_size = window_size
        self.num_points = x_positions.size

        # scale trajectory from px --> µm
        self.x_positions_um = x_positions * self.scale_um  # x(t) [µm]
        self.y_positions_um = y_positions * self.scale_um  # y(t) [µm]
        self.xy_positions_um = np.array([self.x_positions_um, self.y_positions_um]).T

    def compute_motility_metrics(self):
        """Compute motility metrics."""
        # calculate point-to-point distances to facilitate motility measurements
        distances_L1_um = np.diff(self.xy_positions_um, axis=0)
        distances_L2_um = np.linalg.norm(distances_L1_um, axis=1)
        # rolling average to suppress noise
        distances_L2_um_rolling_average = (
            np.convolve(distances_L2_um, np.ones(self.window_size)) / self.window_size
        )
        # rolling sum for max sprint length
        distances_L2_um_cumulative = np.cumsum(distances_L2_um_rolling_average)
        distances_L2_um_rolling_sum = (
            distances_L2_um_cumulative[self.window_size :]
            - distances_L2_um_cumulative[: -self.window_size]
        )

        # instantaneous velocities
        x_velocities = np.diff(self.x_positions_um)
        y_velocities = np.diff(self.y_positions_um)
        # rolling averages for calculating directional change
        x_velocities_rolling_average = (
            np.convolve(x_velocities, np.ones(self.window_size)) / self.window_size
        )
        y_velocities_rolling_average = (
            np.convolve(y_velocities, np.ones(self.window_size)) / self.window_size
        )

        # instantaneous angle and angular change
        angles = np.arctan2(y_velocities_rolling_average, x_velocities_rolling_average)
        # to avoid spikes in the angular change when the angle flips from
        # 2π to 0 we take the minimum angular change by absolute value
        angular_changes = np.stack(
            [np.diff(angles) - 2 * pi, np.diff(angles), np.diff(angles) + 2 * pi]
        ).T
        smallest_change_index = np.abs(angular_changes).argmin(axis=1)
        row_index = np.arange(angular_changes.shape[0])
        angular_change = angular_changes[row_index, smallest_change_index]

        # directional change -- whenever velocity vector flips from
        # positive to negative or vice versa
        x_direction_changes = (
            x_velocities_rolling_average[:-1] * x_velocities_rolling_average[1:] < 0
        )
        y_direction_changes = (
            y_velocities_rolling_average[:-1] * y_velocities_rolling_average[1:] < 0
        )

        # motility measurements
        total_time = (self.num_points - 1) * self.time_increment_s
        total_distance = distances_L2_um.sum()
        net_distance = np.linalg.norm(self.xy_positions_um[-1] - self.xy_positions_um[0])
        max_sprint_length = distances_L2_um_rolling_sum.max()
        confinement_ratio = net_distance / total_distance
        mean_curvilinear_speed = total_distance / total_time
        mean_linear_speed = net_distance / total_time
        mean_angular_speed = np.abs(angular_change).mean() / self.time_increment_s
        num_rotations = np.floor(np.abs(angular_change.sum() / (2 * pi)))
        num_direction_changes = x_direction_changes.sum() + y_direction_changes.sum()
        pivot_rate = num_direction_changes / total_distance

        measurements = {
            "total_time": total_time,
            "total_distance": total_distance,
            "net_distance": net_distance,
            "max_sprint_length": max_sprint_length,
            "confinement_ratio": confinement_ratio,
            "mean_curvilinear_speed": mean_curvilinear_speed,
            "mean_linear_speed": mean_linear_speed,
            "mean_angular_speed": mean_angular_speed,
            "num_rotations": num_rotations,
            "num_direction_changes": num_direction_changes,
            "pivot_rate": pivot_rate,
        }

        return measurements
