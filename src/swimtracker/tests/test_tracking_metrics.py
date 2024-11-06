from math import pi

import numpy as np

from ..tracking_metrics import TrajectoryAnalyzer


class TestTrajectoryAnalyzer:
    def test_compute_motility_metrics_1(self):
        """Simple linear trajectory."""
        time = np.arange(101)
        x_positions = 0.3 * time
        y_positions = -0.4 * time

        trajectory_analyzer = TrajectoryAnalyzer(
            x_positions=x_positions,
            y_positions=y_positions,
        )
        measurements = trajectory_analyzer.compute_motility_metrics()

        assert measurements["total_time"] == 100
        assert measurements["total_distance"] == 50
        assert measurements["net_distance"] == 50
        assert round(measurements["max_sprint_length"], 7) == 0.5
        assert measurements["confinement_ratio"] == 1.0
        assert measurements["mean_curvilinear_speed"] == 0.5
        assert measurements["mean_linear_speed"] == 0.5
        assert round(measurements["mean_angular_speed"], 7) == 0
        assert measurements["num_rotations"] == 0
        assert measurements["num_direction_changes"] == 0
        assert measurements["pivot_rate"] == 0

    def test_compute_motility_metrics_2(self):
        """A short random walk."""
        x_positions = np.array([5, 6, 8, 1, -6, -6, -8, -10, -17, -9, -7, -3])
        y_positions = np.array([8, 14, 14, 20, 16, 8, 10, 3, -5, -1, -6, -3])

        trajectory_analyzer = TrajectoryAnalyzer(
            x_positions=x_positions,
            y_positions=y_positions,
            window_size=1,
        )
        measurements = trajectory_analyzer.compute_motility_metrics()

        np.testing.assert_almost_equal(measurements["total_distance"], 73.4326843)
        np.testing.assert_almost_equal(measurements["net_distance"], 13.6014705)
        np.testing.assert_almost_equal(measurements["max_sprint_length"], 10.6301458)
        np.testing.assert_almost_equal(measurements["confinement_ratio"], 0.1852237)
        np.testing.assert_almost_equal(measurements["mean_curvilinear_speed"], 6.6756986)
        np.testing.assert_almost_equal(measurements["mean_linear_speed"], 1.2364973)
        np.testing.assert_almost_equal(measurements["mean_angular_speed"], 1.7233659)
        np.testing.assert_almost_equal(measurements["num_rotations"], 0.0)
        np.testing.assert_almost_equal(measurements["num_direction_changes"], 8)
        np.testing.assert_almost_equal(measurements["pivot_rate"], 0.1089433)

    def test_compute_motility_metrics_3(self):
        """A clockwise octagon trajectory."""
        time = np.arange(9)
        x_positions = np.cos(time)
        y_positions = -np.sin(time)

        trajectory_analyzer = TrajectoryAnalyzer(
            x_positions=x_positions,
            y_positions=y_positions,
        )
        measurements = trajectory_analyzer.compute_motility_metrics()

        assert measurements["total_time"] == 8
        assert measurements["mean_angular_speed"] == 1
        assert measurements["num_rotations"] == 1
        assert measurements["num_direction_changes"] == 4

    def test_compute_motility_metrics_4(self):
        """A counterclockwise octagon trajectory."""
        time = np.arange(9)
        x_positions = np.cos(time)
        y_positions = np.sin(time)

        trajectory_analyzer = TrajectoryAnalyzer(
            x_positions=x_positions,
            y_positions=y_positions,
        )
        measurements = trajectory_analyzer.compute_motility_metrics()

        assert measurements["total_time"] == 8
        assert measurements["mean_angular_speed"] == 1
        assert measurements["num_rotations"] == 1
        assert measurements["num_direction_changes"] == 4

    def test_compute_motility_metrics_5(self):
        """One period of a sinusoidal trajectory."""
        x_positions = np.linspace(-pi, pi, 51)
        y_positions = np.sin(x_positions)

        trajectory_analyzer = TrajectoryAnalyzer(
            x_positions=x_positions,
            y_positions=y_positions,
        )
        measurements = trajectory_analyzer.compute_motility_metrics()

        np.testing.assert_almost_equal(measurements["total_distance"], 7.6388195)
        np.testing.assert_almost_equal(measurements["net_distance"], 2 * pi)
        np.testing.assert_almost_equal(measurements["max_sprint_length"], 0.1761027)
        np.testing.assert_almost_equal(measurements["confinement_ratio"], 0.8225335)
        np.testing.assert_almost_equal(measurements["mean_curvilinear_speed"], 0.1527764)
        np.testing.assert_almost_equal(measurements["mean_linear_speed"], 0.1256637)
        np.testing.assert_almost_equal(measurements["mean_angular_speed"], 0.0640067)
        np.testing.assert_almost_equal(measurements["num_rotations"], 0.0)
        np.testing.assert_almost_equal(measurements["num_direction_changes"], 2.0)
        np.testing.assert_almost_equal(measurements["pivot_rate"], 0.2618206)

    def test_compute_motility_metrics_6(self):
        """3 clockwise circles followed by 2 counterclockwise circles."""
        # angular velocity
        angular_velocity = 0.01
        # 3 clockwise circles
        time_1 = np.arange(np.ceil(6 * pi / angular_velocity))
        x_positions_1 = np.sin(angular_velocity * time_1)
        y_positions_1 = np.cos(angular_velocity * time_1)
        # 2 counterclockwise circles
        time_2 = np.arange(np.ceil(4 * pi / angular_velocity))
        x_positions_2 = -np.sin(angular_velocity * time_2)
        y_positions_2 = np.cos(angular_velocity * time_2)
        # 3 clockwise circles + 2 counterclockwise circles
        x_positions = np.concatenate([x_positions_1, x_positions_2])
        y_positions = np.concatenate([y_positions_1, y_positions_2])

        trajectory_analyzer = TrajectoryAnalyzer(
            x_positions=x_positions,
            y_positions=y_positions,
        )
        measurements = trajectory_analyzer.compute_motility_metrics()

        np.testing.assert_almost_equal(measurements["total_distance"], 31.4094251)
        np.testing.assert_almost_equal(measurements["net_distance"], 0.0063706)
        np.testing.assert_almost_equal(measurements["max_sprint_length"], angular_velocity)
        np.testing.assert_almost_equal(measurements["mean_curvilinear_speed"], 0.0099998)
        np.testing.assert_almost_equal(measurements["mean_angular_speed"], 0.0109972)
        np.testing.assert_almost_equal(measurements["num_rotations"], 1)
        np.testing.assert_almost_equal(measurements["num_direction_changes"], 20)
        np.testing.assert_almost_equal(measurements["pivot_rate"], 0.6367515)
