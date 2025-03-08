import os

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import src.solutions as solutions
from matplotlib.animation import FuncAnimation
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap


def visualize_object_grid(obj_grid, stage):
    """
    Visualizes a 2x2 grid of object configurations using binary occupancy maps.

    Parameters:
        obj_grids (list of list of ndarray): A list containing four object grid configurations (2D).
        sizes (list of str): A list of four labels corresponding to the object configurations.
    """
    plt.plot(figsize=(3.1, 3.8))

    # grey for spaces not occupied and black for occupied spaces
    cmap = mcolors.ListedColormap(["lightgrey", "black"])

    plt.imshow(obj_grid, cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    plt.title(f"Object Grids (stage = {stage})")
    # plt.savefig("plots/object_layout.png", dpi=300)
    plt.show()


def plot_simulation_without_animation(grid, N, object_grid):
    """
    Generates a visualization of the final state of a grid colorcoded based on concentration.

    Parameters:
        grids (list of numpy.ndarray): 2D concentration grids from the simulation.
        N (int): Grid size (number of spatial points in each, dimension).
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    c_plot = ax.pcolormesh(grid, cmap="viridis", edgecolors="k", linewidth=0.5)

    ax.pcolormesh(object_grid, cmap="Reds", edgecolors="k", linewidth=0.5, alpha=0.5)

    # plt.imshow(c, cmap="viridis", interpolation="nearest", origin="lower")
    plt.colorbar(c_plot, ax=ax, label="Concentration")
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title("2D Diffusion")
    plt.show()


def animate_1a(
    gridd,
    stencill,
    object_gridd,
    grid_indices,
    eta,
    seedje,
    sr_val,
    itertjes=1500,
    interval=0.7,
):
    """
    Generates and saves an animation of a 2D Diffusion-Limited Aggregation (DLA) process.

    The function visualizes the evolution of object placements over time, updating
    the grid based on diffusion and growth probabilities.

    Parameters:
        gridd (numpy.ndarray): 2D diffusion grid representing concentration values.
        stencill (numpy.ndarray): Stencil marking valid growth candidate locations.
        object_gridd (numpy.ndarray): 2D grid indicating object placements.
        grid_indices (numpy.ndarray): Flattened indices of the grid used for random selection.
        eta (float): Growth parameter controlling aggregation probability.
        seedje (int): makes this implementation reproducible.
        sr_val (tuple): Parameters for the SOR update step.
        interval (int, optional): Frame interval in milliseconds (default: 50).
    """
    fig, axs = plt.subplots(figsize=(4.5, 4.5))

    # create colormaps and initial maps
    lilac_purple_cmap = LinearSegmentedColormap.from_list(
        "LilacPurple", ["#440154", "#FFFFFF"]
    )
    img = axs.imshow(gridd, cmap=lilac_purple_cmap, origin="lower", extent=[0, 1, 0, 1])
    plt.colorbar(img, ax=axs, label="Concentration", shrink=0.8)
    object_cmap = mcolors.ListedColormap(["none", "yellow"])  # Only one color, yellow
    object_img = axs.imshow(
        object_gridd, cmap=object_cmap, origin="lower", extent=[0, 1, 0, 1]
    )

    axs.set_title("2D Diffusion Simulation")

    # update grid and viusalize
    def animate(frame):
        nonlocal gridd, stencill, object_gridd, seedje
        seedje += frame
        gridd, object_gridd, stencill, _ = solutions.perform_update_ADL(
            gridd, object_gridd, stencill, grid_indices, eta, seedje, sr_val
        )

        img.set_data(gridd)
        object_img.set_data(object_gridd)
        axs.set_title(f"DLA (η: {eta}) (Step: {frame:.3g})")

        if frame % 100 == 0:
            print(f"finished first {frame} frames")
        return img, object_img

    # Create animation
    animation = FuncAnimation(
        fig, animate, frames=itertjes, interval=interval, blit=False
    )

    animation.save(f"plots/2D_diffusion_p_{eta}.gif", writer="ffmeg", fps=50)
    plt.close(fig)
    return animation


def plot_omega_vs_iterations(omegas, all_iters):
    """
    Plots omega values on the x-axis and the corresponding mean iterations on the y-axis,
    including standard deviation error bars over multiple runs.

    Parameters:
    - all_omega_iters (list of dicts): Each dict contains {omega: total_sor_iters} for a run.
    - omegas (list): List of omega values that were tested.
    """
    fig, axs = plt.subplots(2, 3, figsize=(5, 4), sharey=True, sharex=True)
    axs = axs.flatten()
    iter = 0

    axs[-1].set_axis_off()
    title_string = r"$\eta = $"
    best_omegatjes = dict()

    # Convert list of dictionaries to a structured data format
    for eta, all_omega_iters in all_iters.items():
        omega_iters_matrix = np.array(
            [[run[omega] for omega in omegas] for run in all_omega_iters]
        )

        print(np.shape(omega_iters_matrix))

        # Compute mean and variance within each run
        mean_within_run = np.mean(omega_iters_matrix, axis=2)
        var_within_run = np.var(omega_iters_matrix, axis=2)

        # Compute mean of means between runs
        mean_of_means = np.mean(mean_within_run, axis=0)

        # Compute variance between runs
        var_between_runs = np.var(mean_within_run, axis=0)

        # Compute total variance (sum of between-run variance and within-run variance)
        total_variance = var_between_runs + np.mean(var_within_run, axis=0)

        min_iters = np.argmin(mean_of_means)
        best_omega = omegas[min_iters]
        best_omegatjes[eta] = best_omega

        # plot mean with errorbar
        axs[iter].errorbar(
            omegas,
            mean_of_means,
            yerr=np.sqrt(total_variance),
            fmt="o-",
            capsize=5,
            label=f"η: {eta}",
            color="#440154",
            alpha=1,
        )
        axs[iter].set_title(title_string + f"{eta} ")
        axs[iter].grid(True)

        iter += 1

    axs[2].xaxis.set_tick_params(which="both", labelbottom=True)
    axs[0].set_ylabel("Iterations")
    axs[3].set_ylabel("Iterations")
    for j in [2, 3, 4]:
        axs[j].set_xlabel(r"$\omega$")

    plt.suptitle(r"Effect of $\omega$ on Iterations for SOR (DLA)")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.23)
    plt.savefig("plots/opt_omega_DLA.png", dpi=300)
    plt.show()
    return best_omegatjes


def plot_five_DLA(gridjes, etas):
    """
    Visualizes the evolution of a 2D DLA process at different η values.

    Parameters:
        gridjes (dict): Dictionary mapping each η value to a tuple (grid, object_grid), where:
            - grid (numpy.ndarray): 2D array representing concentration values.
            - object_grid (numpy.ndarray): 2D array indicating object placements.
        etas (list of float): List of η values for which the diffusion snapshots will be plotted.

    """
    # plot setup
    assert len(etas) == 5, (
        f"The setup of this plot is for 5 different eta values, not {len(etas)}"
    )
    fig, axs = plt.subplots(2, 3, figsize=(5.2, 4), sharex=True, sharey=True)
    fig.suptitle(r"DLA Process for Different $\eta$ Values ")
    axs = axs.flatten()

    # colormaps
    lilac_purple_cmap = LinearSegmentedColormap.from_list(
        "LilacPurple", ["#440154", "#FFFFFF"]
    )
    object_cmap = mcolors.ListedColormap(["none", "yellow"])  # Only one color, yellow

    # Hide the last unused subplot
    axs[-1].set_visible(False)
    for i, e in enumerate(etas):
        gridd, object_gridd = gridjes[e]
        img = axs[i].imshow(
            gridd, cmap=lilac_purple_cmap, origin="lower", extent=[0, 1, 0, 1]
        )
        axs[i].imshow(
            object_gridd, cmap=object_cmap, origin="lower", extent=[0, 1, 0, 1]
        )
        axs[i].set_title(r"$\eta:$" + f" {e}")
        if i > 1:
            axs[i].set_xlabel("x")

    # set proper ticks and labels
    axs[2].xaxis.set_tick_params(which="both", labelbottom=True)
    axs[0].set_ylabel("y")
    axs[3].set_ylabel("y")

    cbar_ax = fig.add_axes([0.92, 0.09, 0.02, 0.7])  # [left, bottom, width, height]
    # fig.colorbar(im, cax=cbar_ax, label="Concentration")
    fig.colorbar(img, cax=cbar_ax, label="Concentration", shrink=0.8)

    plt.tight_layout(rect=[0, 0, 0.93, 1])
    plt.subplots_adjust(wspace=0.22, hspace=0.09)
    plt.savefig("plots/DLA_snapshots.png", dpi=300, bbox_inches="tight")
    plt.show()


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

    colors = ["#440154", "yellow", "#4A90E2"]
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots()
    fig.suptitle("DLA using Monte Carlo Simulations", fontsize=16)
    img = ax.imshow(all_grids[0], cmap=cmap, origin="lower")

    skip = 5

    def update(frame):
        img.set_data(all_grids[frame])
        ax.set_title(f"Step {frame * skip}")

    anim = FuncAnimation(
        fig, update, frames=range(0, len(all_grids), skip), blit=False, interval=5
    )
    plt.close()

    # anim.save("plots/animation_random_walker.gif", writer="pillow", dpi=300)
    return anim


def visualize_for_different_probs(grids, probs_join):
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
        axs[i].set_title(r"$p_s = {}$".format(probs_join[i]))
        if i > 1:
            axs[i].set_xlabel("x")

    # set proper ticks and labels
    axs[2].xaxis.set_tick_params(which="both", labelbottom=True)
    axs[0].set_ylabel("y")
    axs[3].set_ylabel("y")

    legend_patches = [
        mpatches.Patch(color=colors[0], label="Empty Cells"),
        mpatches.Patch(color=colors[1], label="Cluster Cells"),
        mpatches.Patch(color=colors[2], label="Random Walkers"),
    ]

    fig.legend(
        handles=legend_patches,
        loc="lower right",
        bbox_to_anchor=(0.97, 0.19),
        fontsize=8,
        title="Categories",
        markerscale=0.8,
    )
    # plt.tight_layout()

    plt.subplots_adjust(wspace=0.22, hspace=0.2)
    plt.savefig(
        "plots/random_walker_snapshots.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_final_gray_scott(u_final_list, param_sets, N, output_dir="plots"):
    """
    Generates and saves a 2x2 grid of final concentration fields from the Gray-Scott model.

     Parameters:
         u_final_list (list): List of 2D numpy arrays representing final U concentrations.
         param_sets (list): List of (f, k) parameter tuples.
         N (int): Grid size.
         output_dir (string): Directory to save the output figure is 'plots'.
    """
    os.makedirs(output_dir, exist_ok=True)

    # colormap
    cmap = LinearSegmentedColormap.from_list(
        "custom_colormap", ["#440154", "#3b528b", "#fde725"], N=256
    )

    fig, axs = plt.subplots(2, 2, figsize=(3.5, 3.9), sharex=True, sharey=True)
    fig.suptitle(r"Final U concentration for varying $f, k$")
    axs = axs.flatten()

    # iterate over parameter sets
    for idx, (ax, u_final, param_set) in enumerate(zip(axs, u_final_list, param_sets)):
        img = ax.imshow(
            u_final, cmap=cmap, origin="lower", vmin=0.3, vmax=1, extent=[0, N, 0, N]
        )
        ax.set_title(
            r"$f$: " + f"{param_set[0]:.3f}, " + r"$k$: " + f"{param_set[1]:.3f}",
            fontsize=10,
        )

    # y-axis (labels and ticks)
    axs[0].set_ylabel("y")
    axs[2].set_ylabel("y")
    axs[0].set_yticks([0, N // 2, N])
    axs[2].set_yticks([0, N // 2, N])

    # x-axis (labels and ticks)
    axs[2].xaxis.set_tick_params(which="both", labelbottom=True)
    axs[3].xaxis.set_tick_params(which="both", labelbottom=True)
    axs[2].set_xlabel("x")
    axs[3].set_xlabel("x")
    axs[2].set_xticks([0, N // 2, N])
    axs[3].set_xticks([0, N // 2, N])

    plt.subplots_adjust(wspace=0.25, hspace=0.25)

    # colorbar
    cbar_ax = fig.add_axes(
        [0.95, 0.12, 0.03, 0.74]
    )  # [<left>, <bottom>, <width>, <height>]
    cbar = fig.colorbar(img, cax=cbar_ax, shrink=0.8)
    cbar.set_label("Concentration of U", fontsize=12)

    frame_path = f"{output_dir}/gray_scott_multi.png"
    plt.savefig(frame_path, bbox_inches="tight", dpi=300)
    plt.show()
