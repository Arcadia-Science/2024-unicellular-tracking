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

    def measure_trajectories(self):
        """

        Returns
        -------
        dataframe : pd.DataFrame
        """
        if not hasattr(self, "frametime"):
            msg = (
                "Must initialize `TrajectoryCSVParser` with a frametime in "
                "order to make trajectory measurements."
            )
            raise AttributeError(msg)

    def get_trajectory_count(self):
        """"""
        return self.dataframe["ID"].unique().size

    def estimate_cell_count(self):
        """"""
        num_cells = 3

        return num_cells


class TrajectoryAnalyzer:
    """Class for analyzing cell trajectories.

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
    [1]
    """

    def __init__(self, x, y, dt) -> None:
        self.x = x
        self.y = y
        self.dt = dt
        self.points = np.array([x, y]).T

        # calculate instantaneous_quantities to facilitate motility measurements
        self._calculate_instantaneous_quantities()

        # motility measurements
        t_tot = self.calculate_total_time()
        d_tot = self.calculate_total_distance()
        d_net = self.calculate_net_distance()
        r_con = self.calculate_confinement_ratio()
        v_crv = self.calculate_mean_curvilinear_speed()
        v_lin = self.calculate_mean_linear_speed()
        r_lin = self.calculate_linearity_of_progression()

        self.measurements = {
            "total_time": t_tot,
            "total_distance": d_tot,
            "net_distance": d_net,
            "confinement_ratio": r_con,
            "mean_curvilinear_speed": v_crv,
            "mean_linear_speed": v_lin,
            "linearity_of_progression": r_lin,
        }

    def _calculate_instantaneous_quantities(self):
        """"""
        # instantaneous distances
        self.L1_distances = np.diff(self.points, axis=0)
        self.L2_distances = np.linalg.norm(self.L1_distances, axis=1)

        # instantaneous angle
        dx = np.diff(self.x)
        dy = np.diff(self.y)
        self.alpha = np.arctan2(dy, dx)
        # instantaneous directional change
        self.gamma = np.cumsum(np.abs(self.alpha))

        # instantaneous linear and angular velocity
        self.linear_velocity = self.L2_distances / self.dt
        self.angular_velocity = self.gamma / self.dt

    def calculate_total_time(self):
        """Calculate the total time in which the cell traveled.

        Equation
        --------
        t_tot = N * dt
        """
        return self.x * self.dt

    def calculate_total_distance(self):
        """Calculate the total distance traveled by the cell.

        Equation
        --------
        d_tot = sum( distance(p_i, p_i+1) ) for i = 0 --> N
        """
        return self.L2_distances.sum()

    def calculate_net_distance(self):
        """Calculate the net distance traveled by the cell.

        Equation
        --------
        d_net = distance(p_0, p_N)
        """
        return np.linalg.norm(self.points[-1] - self.points[0])

    def calculate_confinement_ratio(self):
        """Calculate the confinement ratio.

        Equation
        --------
        r_con = d_net / d_tot
        """
        return self.net_distance / self.total_distance

    def calculate_mean_curvilinear_speed(self):
        """Calculate the average speed along the curved trajectory.

        Equation
        --------
        v_avg = 1 / N * sum( v_i ) for i = 0 --> N
        """
        return self.linear_velocity.mean()

    def calculate_mean_linear_speed(self):
        """Calculate the average straight line speed.

        Equation
        --------
        v_lin = d_net / t_tot
        """
        return self.net_distance / self.total_time

    def calculate_linearity_of_progression(self):
        """Calculate the linearity of forward progression.

        Equation
        --------
        r_lin = v_lin / v_avg
        """
        return self.mean_linear_speed / self.mean_curvilinear_speed

    def to_dataframe(self):
        """"""
        dataframe = pd.DataFrame()
        return dataframe
