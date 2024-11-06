import numpy as np
import seaborn as sns
from scipy import stats


def map_p_value_to_asterisks(p_value):
    """Map a p-value to a symbol representing statistical significance."""
    if p_value <= 0.0001:
        return "****"
    elif p_value <= 0.001:
        return "***"
    elif p_value <= 0.01:
        return "**"
    elif p_value <= 0.05:
        return "*"
    else:
        return "ns"


def ensure_two_groups(data, groupby_variable):
    """Ensure that data is split into two distinct groups for applying statistical tests."""
    num_groups = data.groupby(groupby_variable).ngroups
    if num_groups != 2:
        msg = (
            "This function only supports statistical tests for two distinct groupings, but "
            f"{num_groups} were provided."
        )
        raise ValueError(msg)


def split_violin_plot_with_stats(
    data,
    x_variable,
    y_variable,
    hue_variable,
    min_sample_size=6,
    **split_violin_kwargs,
):
    """Run statistical tests and add annotations of the results to a seaborn violin plot."""
    data = data.reset_index(drop=True).copy()
    ensure_two_groups(data, groupby_variable=hue_variable)

    # render violin plot
    ax = sns.violinplot(
        data=data,
        x=x_variable,
        y=y_variable,
        hue=hue_variable,
        split=True,
        **split_violin_kwargs,
    )

    # extract distributions for statistical tests
    group_keys = []
    group_x_values = []
    for key, group in data.groupby(hue_variable):
        group_keys.append(key)
        group_x_values.append(group[y_variable].values)

    # annotate stats
    annotate_statistical_significance(
        *group_x_values,
        ax.axes,
        min_sample_size,
        orientation="horizontal",
        center_annotation=True,
    )

    # add sample size to legend
    if ax.axes.get_legend() is not None:
        for legend_label in ax.axes.get_legend().texts:
            init_label = legend_label.get_text()
            group_key = next(k for k in group_keys if str(k) == init_label)
            sample_size = len(data.groupby(hue_variable).get_group(group_key))
            legend_label.set_text(f"{init_label} | n={sample_size}")

    return ax


def joint_grid_with_stats(
    data,
    x_variable,
    y_variable,
    hue_variable,
    min_sample_size=6,
    **joint_grid_kwargs,
):
    """Run statistical tests and add annotations of the results to a seaborn JointGrid."""
    data = data.reset_index(drop=True).copy()
    ensure_two_groups(data, groupby_variable=hue_variable)

    # render JointGrid
    ax = sns.JointGrid(
        data=data,
        x=x_variable,
        y=y_variable,
        hue=hue_variable,
        **joint_grid_kwargs,
    )

    # plot onto JointGrid
    ax.plot_joint(sns.scatterplot, legend=True)
    ax.plot_marginals(sns.kdeplot, lw=2, fill=True, common_norm=False)

    # extract distributions for statistical tests
    group_keys = []
    group_x_values = []
    group_y_values = []
    for key, group in data.groupby(hue_variable):
        group_keys.append(key)
        group_x_values.append(group[x_variable].values)
        group_y_values.append(group[y_variable].values)

    # annotate stats
    annotate_statistical_significance(
        *group_x_values, ax.ax_marg_x, min_sample_size, orientation="horizontal"
    )
    annotate_statistical_significance(
        *group_y_values, ax.ax_marg_y, min_sample_size, orientation="vertical"
    )

    # add sample size to legend
    if ax.ax_joint.get_legend() is not None:
        for legend_label in ax.ax_joint.get_legend().texts:
            init_label = legend_label.get_text()
            group_key = next(k for k in group_keys if str(k) == init_label)
            sample_size = len(data.groupby(hue_variable).get_group(group_key))
            legend_label.set_text(f"{init_label} | n={sample_size}")

    return ax


def annotate_statistical_significance(
    sample_a_values,
    sample_b_values,
    matplotlib_axis,
    min_sample_size=6,
    center_annotation=False,
    orientation="horizontal",
):
    """Measure statistical significance of two distributions and annotate plot accordingly.

    Statistical significance is measured using the Mann-Whitney U test [1], which is a versatile
    nonparametric test suitable for comparing two independent samples, particularly when the data
    does not meet the assumptions required for parametric tests. Input distributions should be
    tested for normality by e.g. D'Agostino's K-squared test [2] prior to running this function.

    Parameters
    ----------
    sample_a_values, sample_b_values : (N,) array-like
        Distributions for the statistical test from two samples "a" and "b".
    matplotlib_axis : `matplotlib.axes.Axes`
        Matplotlib axis to annotate.
    center_annotation : bool
        Whether to place the annotation in the center of the axis. Appropriate for when the x-axis
        is categorical as opposed to numerical.
    orientation : str
        Whether the matplotlib axis is oriented horizontally or vertically.

    References
    ----------
    [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
    [2] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html
    """
    if (sample_a_values.size < min_sample_size) or (sample_b_values.size < min_sample_size):
        msg = "Sample size of one or both distributions less than `min_sample_size`."
        raise ValueError(msg)

    # Mann-Whitney U test
    _, p_value = stats.mannwhitneyu(sample_a_values, sample_b_values, alternative="two-sided")
    # get appropriate number of asterisks based on p-value
    significance_text = map_p_value_to_asterisks(p_value)

    # create a horizontal or vertical bar below either a series of asterisks or "ns"
    fontsize = 20 if "*" in significance_text else 16
    if orientation == "horizontal":
        text_padding = 0 if "*" in significance_text else 0.05
        x_start, x_center, x_end, y_center = _get_coordinates_for_annotation(
            sample_a_values, sample_b_values, matplotlib_axis, center_annotation, orientation
        )
        matplotlib_axis.plot([x_start, x_end], [y_center, y_center], "k-")
        matplotlib_axis.text(
            x_center,
            (1 + text_padding) * y_center,
            significance_text,
            fontsize=fontsize,
            ha="center",
        )

    elif orientation == "vertical":
        text_padding = -0.2 if "*" in significance_text else 0.05
        y_start, y_center, y_end, x_center = _get_coordinates_for_annotation(
            sample_a_values, sample_b_values, matplotlib_axis, center_annotation, orientation
        )
        matplotlib_axis.plot([x_center, x_center], [y_start, y_end], "k-")
        matplotlib_axis.text(
            (1 + text_padding) * x_center,
            y_center,
            significance_text,
            fontsize=fontsize,
            va="center",
            rotation=270,
        )

    else:
        msg = f"Unknown orientation '{orientation}'."
        raise ValueError(msg)


def _get_coordinates_for_annotation(
    sample_a_values,
    sample_b_values,
    matplotlib_axis,
    center_annotation,
    orientation,
):
    """Return axes coordinates for drawing the statistical annotation."""
    x_limits = matplotlib_axis.get_xlim()
    y_limits = matplotlib_axis.get_ylim()
    main_axis_limits = x_limits if orientation == "horizontal" else y_limits
    cross_axis_limits = y_limits if orientation == "horizontal" else x_limits

    if center_annotation:
        main_axis_center = 0.5
        main_axis_start, main_axis_end = 0.3, 0.7
    else:
        range = main_axis_limits[1] - main_axis_limits[0]
        main_axis_center = (np.median(sample_a_values) + np.median(sample_b_values)) / 2
        main_axis_start = main_axis_center - range / 8
        main_axis_end = main_axis_center + range / 8

    # the cross-axis coordinate is independent of plotting style.
    cross_axis_center = 1.2 * cross_axis_limits[1]

    return main_axis_start, main_axis_center, main_axis_end, cross_axis_center
