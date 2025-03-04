import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import BoundaryNorm, ListedColormap


def animate_mc_dla(all_grids):
    """
    Animate the Monte Carlo Diffusion-Limited Aggregation (DLA) simulation.

    Parameters:
        all_grids (list of numpy.ndarray): A list of 2D numpy arrays representing the state of the grid
                                           at each simulation step. These arrays are used sequentially to
                                           generate the animation frames.

    Returns:
        matplotlib.animation.FuncAnimation: The animation object representing the DLA simulation animation.
    """

    fig, ax = plt.subplots()
    img = ax.imshow(all_grids[0], cmap="viridis", origin="lower")

    def update(frame):
        img.set_data(all_grids[frame])
        ax.set_title(f"Snapchat {frame} (step {frame * 10})")

    anim = FuncAnimation(
        fig, update, frames=range(len(all_grids)), blit=False, interval=16
    )
    anim.save("plots/animation_random_walker.gif", writer="pillow", dpi=300)
    plt.close(fig)

    return anim


def animate_for_different_probs(grids, probs_join):
    """
    Create a subplot figure to visualize grid snapshots for different joining probabilities.

    Parameters:
        grids (list of numpy.ndarray): A list of 2D arrays representing grid snapshots of the simulation.
                                       The function uses the first five elements in this list.
        probs_join (list): A list of joining probability values corresponding to each grid snapshot in 'grids'.

    Returns:
        None
    """

    fig, axs = plt.subplots(2, 3, figsize=(4.4, 3.4), sharex=True, sharey=True)
    fig.suptitle("Cluster Sizes For 'Sticky' Probabilities")
    axs = axs.flatten()

    # purple, blue, yellow
    colors = ["#440154", "yellow", "#4A90E2"]
    cmap = ListedColormap(colors)
    bounds = [0, 1, 2, 3]
    norm = BoundaryNorm(bounds, ncolors=cmap.N)

    # Hide the last unused subplot
    axs[-1].set_visible(False)
    for i in range(5):
        im = axs[i].imshow(
            grids[i],
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            origin="lower",
            extent=[0, 1, 0, 1],
        )
        axs[i].set_title(f" Prob = {probs_join[i]}")
        if i > 1:
            axs[i].set_xlabel("x")

    # set proper ticks and labels
    axs[2].xaxis.set_tick_params(which="both", labelbottom=True)
    axs[0].set_ylabel("y")
    axs[3].set_ylabel("y")

    cbar_ax = fig.add_axes([0.94, 0.09, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, label="Concentration")

    # fig.legend(handles=legend_elements, loc="lower right", bbox_to_anchor=(1.1, 1.1))
    # plt.tight_layout()
    plt.subplots_adjust(wspace=0.22, hspace=0.2)
    plt.savefig(
        "plots/random_walker_snapshots.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
