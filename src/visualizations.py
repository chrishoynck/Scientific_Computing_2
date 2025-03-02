import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import src.solutions as solutions
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation

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

def animate_1a(gridd, stencill, object_gridd, grid_indices, eta, seedje, sr_val, itertjes=1500, interval=0.7):
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

    lilac_purple_cmap = LinearSegmentedColormap.from_list("LilacPurple", ["#440154", "#FFFFFF"])
    img = axs.imshow(gridd, cmap=lilac_purple_cmap, origin="lower", extent=[0, 1, 0, 1])
    plt.colorbar(img, ax=axs, label="Concentration", shrink=0.8)

    # object_mask = np.zeros_like(object_gridd, dtype=float)
    # object_mask[object_gridd ==1] = 1  # Mark object positions
    object_cmap = mcolors.ListedColormap(["none", "yellow"])  # Only one color, yellow
    object_img = axs.imshow(object_gridd, cmap=object_cmap, origin="lower", extent=[0, 1, 0, 1])
    
    # img = axs.imshow(object_gridd, cmap=cmap, origin="lower", extent=[0, 1, 0, 1])
    # plt.colorbar(img, ax=axs, label="Concentration", shrink=0.8)
    axs.set_title("2D Diffusion Simulation")

    def animate(frame):
        nonlocal  gridd, stencill, object_gridd, seedje
        seedje += frame
        gridd, object_gridd, stencill, _ = solutions.perform_update_ADL(gridd, object_gridd, stencill, grid_indices, eta, seedje, sr_val)
        
        img.set_data(gridd)
        object_img.set_data(object_gridd)
        axs.set_title(f"DLA (η: {eta}) (Step: {frame:.3g})")

        if frame%100 == 0:
            print(f"finished first {frame} frames")
        return img, object_img
    # Create animation
    animation = FuncAnimation(
        fig, animate, frames=itertjes, interval=interval, blit=False
    )
    # writer=PillowWriter(fps=200, metadata={"duration": 0.05})
    animation.save(f"plots/2D_diffusion_p_{eta}.gif", writer="ffmeg", fps=50)

    plt.close(fig)
    return animation


def plot_five_DLA(gridd, object_gridd, etas):
    """
    Visualizes the evolution of a 2D diffusion process at five different time points.

    Generates a grid of subplots displaying the concentration distribution at 
    t = 0, 0.001, 0.01, 0.1, 1.0. 

    Parameters:
        all_c (list of numpy.ndarray): 2D concentration grids at specified time points.
        times (list of float): Time points corresponding to the concentration grids.
    """
    # plot setup
    fig, axs = plt.subplots(2, 3, figsize=(4.4, 3.4), sharex=True, sharey=True)
    fig.suptitle("2D Diffusion at Different t Values")
    axs = axs.flatten()

    lilac_purple_cmap = LinearSegmentedColormap.from_list("LilacPurple", ["#440154", "#FFFFFF"])
    object_cmap = mcolors.ListedColormap(["none", "yellow"])  # Only one color, yellow
    
    # Hide the last unused subplot
    axs[-1].set_visible(False)
    for i in range(5):
        
        img = axs[i].imshow(gridd, cmap=lilac_purple_cmap, origin="lower", extent=[0, 1, 0, 1])
        object_img = axs.imshow(object_gridd, cmap=object_cmap, origin="lower", extent=[0, 1, 0, 1])
        axs[i].set_title(r"$\eta:$" + f" {etas[i]}")
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
    plt.subplots_adjust(wspace=0.22, hspace=0.2)
    plt.savefig("plots/diffusion_snapshots.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_omega_vs_iterations(eta_list, omegas):
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
    title_string = r'$\eta = $'
    best_omegatjes = dict()

    # Convert list of dictionaries to a structured data format
    for eta, all_omega_iters in eta_list.items():
        # all_runs.extend(all_omega_iters)
        omega_iters_matrix = np.array([[run[omega] for omega in omegas] for run in all_omega_iters])

        # Compute mean and standard deviation
        mean_iters = np.mean(omega_iters_matrix, axis=0)
        std_iters = np.std(omega_iters_matrix, axis=0)
        
        # establish optimal omega for specific eta
        min_iters = np.argmin(mean_iters)
        best_omega = omegas[min_iters]
        best_omegatjes[eta] = best_omega

        # plot mean with errorbar 
        axs[iter].errorbar(omegas, mean_iters, yerr=std_iters, fmt='o-', capsize=5, label=f"η: {eta}", alpha=0.7)
        axs[iter].set_title(title_string + f"{eta} ")
        axs[iter].grid(True)

        iter +=1

    axs[2].xaxis.set_tick_params(which="both", labelbottom=True)
    axs[0].set_ylabel('Iterations')
    axs[3].set_ylabel('Iterations')
    for j in [2, 3, 4]:
        axs[j].set_xlabel(r'$\omega$')
    
    plt.suptitle(r'Effect of $\omega$ on Iterations for SOR (DLA)')
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
    fig, axs = plt.subplots(2, 3, figsize=(5.2, 4), sharex=True, sharey=True)
    fig.suptitle(r"DLA Process for Different $\eta$ Values ")
    axs = axs.flatten()

    lilac_purple_cmap = LinearSegmentedColormap.from_list("LilacPurple", ["#440154", "#FFFFFF"])
    object_cmap = mcolors.ListedColormap(["none", "yellow"])  # Only one color, yellow
    
    # Hide the last unused subplot
    axs[-1].set_visible(False)
    for i, e in enumerate(etas):
        gridd, object_gridd = gridjes[e]
        img = axs[i].imshow(gridd, cmap=lilac_purple_cmap, origin="lower", extent=[0, 1, 0, 1])
        object_img = axs[i].imshow(object_gridd, cmap=object_cmap, origin="lower", extent=[0, 1, 0, 1])
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