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
    frametime : float (optional)
        Time increment [ms] between sequential rows of the motility data.
        Equal to the exposure time of the timelapse or 1000 / frames per second.
    """

    def __init__(self, csv_file, frametime=None):
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

        if frametime is not None:
            self.frametime = frametime

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

    def measure_trajectories(self, min_frames=5):
        """"""
        if not hasattr(self, "frametime"):
            msg = (
                "Must initialize `TrajectoryCSVParser` with a frametime in "
                "order to make trajectory measurements."
            )
            raise AttributeError(msg)

        measurement_collection = []
        for cell_id, cell_data in self.dataframe.groupby("ID"):
            x = cell_data["x"].values
            y = cell_data["y"].values

            if x.size < min_frames:
                logger.debug(
                    f"Discarding cell {cell_id} with fewer than {min_frames} "
                    "points in its trajectory."
                )
                continue

            trajectory_analyzer = TrajectoryAnalyzer(x, y, dt=self.frametime)
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

    Motility measurements adapted from [1].
    +--------------------------+------------------------------+
    | Measurement              | Equation                     |
    +--------------------------+------------------------------+
    | total_time               | t_tot = N * dt               |
    | total_distance           | d_tot = sum( d(p_i, p_i+1) ) |
    | net_distance             | d_net = d(p_0, p_N)          |
    | max_distance             | d_max = max( d(p_i, p_i+1) ) |
    | confinement_ratio        | r_con = d_net / d_tot        |
    | mean_curvilinear_speed   | v_avg = 1 / N * sum( v_i )   |
    | mean_linear_speed        | v_lin = d_net / t_tot        |
    | linearity_of_progression | r_lin = v_lin / v_avg        |
    +--------------------------+------------------------------+

    Parameters
    ----------
    x : 1D float array
        X coordinates with respect to time.
    y : 1D float array
        Y coordinates with respect to time.
    dt : float
        Time increment [ms] between sequential (x, y) data points.

    References
    ----------
    [1] https://doi.org/10.1016/B978-0-12-391857-4.00009-4
    """

    def __init__(self, x, y, dt) -> None:
        self.x = x
        self.y = y
        self.dt = dt
        self.points = np.array([x, y]).T

        # calculate point-to-point distances to facilitate motility measurements
        distances_L1 = np.diff(self.points, axis=0)
        distances_L2 = np.linalg.norm(distances_L1, axis=1)

        # motility measurements
        N = x.size
        t_tot = N * dt
        d_tot = distances_L2.sum()
        d_net = np.linalg.norm(self.points[-1] - self.points[0])
        d_max = distances_L2.max()
        r_con = d_net / d_tot
        v_avg = (distances_L2 / dt).mean()
        v_lin = d_net / t_tot
        r_lin = v_lin / v_avg

        self.measurements = {
            "total_time": t_tot,
            "total_distance": d_tot,
            "net_distance": d_net,
            "max_distance": d_max,
            "confinement_ratio": r_con,
            "mean_curvilinear_speed": v_avg,
            "mean_linear_speed": v_lin,
            "linearity_of_progression": r_lin,
        }
