import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


def plot_stack_as_cartoon(
    stack,
    skip=20,
    figsize=(8, 8)
):
    """Plot every `skip` frames of `stack` like cartoon panels."""
    cartoon = ski.util.montage(
        stack[::skip],
        padding_width=4,
        fill=0
    )
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(cartoon, cmap="Greys_r")
    ax.set_axis_off()


def plot_extracted_pools(finder):
    """Plotting function for visualizing pool extraction.

    Parameters
    ----------
    finder : `PoolFinder` instance

    Layout
    ------
    upper left
        Mean projection of timelapse annotated to highlight extracted pools.
    upper right
        Debug image (see `PoolFinder.make_debug_sketch`).
    bottom
        Each subplot is a mean projection image of an extracted pool.
    """

    # create figure with `matplotlib.GridSpec`
    ncols = 4
    nrows = 4
    fig = plt.figure(
        constrained_layout=True,
        figsize=(2*ncols, 2*nrows)
    )
    fig.suptitle(finder.filepath.stem)
    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols,
    )

    ax_proj = fig.add_subplot(gs[:2, :2])
    ax_proj.imshow(finder.mean_intensity_projection, cmap="Greys_r")

    # plot debug image
    dbug = finder.make_debug_sketch()
    ax_dbug = fig.add_subplot(gs[:2, 2:])
    ax_dbug.imshow(dbug)

    # plot first 8 pools
    for i, ((ix, iy), pool) in enumerate(finder.pools.items()):

        # compute std projection of pool and plot
        nz, *_ = pool.stack_raw.shape
        image = pool.stack_raw.std(axis=0)
        ax = fig.add_subplot(gs[i//4 + 2, i % 4])
        ax.imshow(image, cmap="Greys_r")
        title = f"Pool ({ix}, {iy})\nHas cells: {pool.has_cells()}"
        ax.set_title(title)

        # annotate projection image
        center = finder.poolmap[(ix, iy)][0]
        rect = Rectangle(
            xy=(
                center[0] - finder.pool_radius_px,
                center[1] - finder.pool_radius_px
            ),
            width=2*finder.pool_radius_px,
            height=2*finder.pool_radius_px,
            facecolor="none",
            edgecolor="#22ffff"
        )

        # annotate pools in mean intensity projection image
        ax_proj.add_patch(rect)
        ax_proj.text(
            x=center[0],
            y=center[1] - finder.pool_radius_px - 2,
            s=f"({ix}, {iy})",
            color="#22ffff",
            ha="center",
            va="bottom"
        )

        # increment and stop condition
        i += 1
        if i > 7:
            break


def get_neon_cmap():
    """Create a colormap with super bright colors above a black background."""
    # bright colors
    base_colors = [
        '#000000',  # black
        '#ff99aa',  # pink
        '#ffcc00',  # orange
        '#ffff00',  # yellow
        '#00ff00',  # green
        '#00ddff',  # cyan
    ]
    # create matplotlib colormap from colors
    weights = [0] + np.linspace(0, 1, len(base_colors)-1).tolist()
    weighted_colors = list(zip(weights, base_colors, strict=False))
    cmap = LinearSegmentedColormap.from_list("neon", weighted_colors)
    return cmap
