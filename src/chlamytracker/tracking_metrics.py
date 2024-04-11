import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TrajectoryCSVParser:
    """Class for tracking cell motility within agar microchamber pools.

    Parameters
    ----------
    csv_file : str or `pathlib.Path`
        Path to csv file for analyzing.
    """

    def __init__(self, csv_file):

        # this is annoying but basically I output a ton of data with one particular
        # csv format. there are a number of small things that distinguish them but
        # the main one is that one is actually comma delimited whereas the other
        # is white space delimited. so this is how I will differentiate them for now.
        with open(csv_file) as csv:
            csv_headers = next(csv)

        if len(csv_headers.split(" ")) > 1:
            properties = self.set_properties()
            self.dataframe = pd.read_csv(csv_file, sep="\\s+", names=properties, skiprows=1)
        elif len(csv_headers.split(",")) > 1:
            self.dataframe = pd.read_csv(csv_file)
        else:
            raise ValueError(f"Unable to read csv file {csv_file}.")

        self.trajectory_count = self.get_trajectory_count()
        self.cell_count = self.estimate_cell_count()

    def set_properties(self):
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

        properties = default_btrack_csv_columns + default_base_properties
        return properties

    def get_trajectories(self):
        """"""
        cell_trajectories = []
        for _cell_id, cell_data in self.dataframe.groupby("ID"):
            x = cell_data["x"].values
            y = cell_data["y"].values
            trajectory = TrajectoryAnalyzer(x, y)
            cell_trajectories.append(trajectory)

        return cell_trajectories

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

    References
    ----------
    [1]
    """

    def __init__(self, x, y) -> None:
        self.points = np.array([x, y]).T

        # # instantaneous velocities
        # vx = np.diff(x)
        # vy = np.diff(y)

        # instantaneous distances
        self.distances_L1 = np.diff(self.points, axis=0)
        self.distances_L2 = np.linalg.norm(self.distances_L1, axis=1)

    #     # instantaneous angle
    #     alpha = np.arctan2(vy, vx)
    #     # instantaneous directional change
    #     gamma = 3

    #     self.total_distance_traveled = self.calculate_total_distance_traveled()
    #     self.net_distance_traveled = self.calculate_net_distance_traveled()
    #     self.confinement_ratio = self.calculate_confinement_ratio()
    #     self.mean_curvilinear_speed = self.calculate_mean_curvilinear_speed()
    #     self.mean_linear_speed = self.calculate_mean_linear_speed()
    #     self.linearity_of_progression = self.calculate_linearity_of_progression()
    #     self.mean_squared_displacement = self.calculate_mean_squared_displacement()

    # def calculate_total_distance_traveled(self):
    #     """Calculate the total distance traveled by the cell.

    #     Equation
    #     --------
    #     d_tot = sum[distance(p_i --> p_i+1)] from i = 0-->N
    #     """
    #     return self.distances_L2.sum()

    # def calculate_net_distance_traveled(self):
    #     """Calculate the net distance traveled by the cell.

    #     Equation
    #     --------
    #     d_net = distance(p_0, p_N)
    #     """
    #     net_distance = np.linalg.norm(self.points[-1] - self.points[0])
    #     return net_distance

    # def calculate_confinement_ratio(self):
    #     """Calculate the confinement ratio.

    #     Equation
    #     --------
    #     r_con = d_net / d_tot
    #     """
    #     return self.net_distance_traveled / self.total_distance_traveled

    # def calculate_instantaneous_angle(self):
    #     """"""
    #     pass

    # def calculate_directional_change(self):
    #     """"""
    #     pass

    # def calculate_instantaneous_speed(self):
    #     """"""
    #     pass

    # def calculate_mean_curvilinear_speed(self):
    #     """"""
    #     pass

    # def calculate_mean_linear_speed(self):
    #     """"""
    #     pass

    # def calculate_linearity_of_progression(self):
    #     """"""
    #     pass

    # def calculate_mean_squared_displacement(self):
    #     """"""
    #     pass

    # # def from_csv()
    # # parse csv
    # # return x, y
