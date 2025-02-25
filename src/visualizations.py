import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import src.solutions as solutions
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

def animate_1a(gridd, stencill, object_gridd, grid_indices, eta, seedje, sr_val, interval=50):
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

    cmap = mcolors.ListedColormap(["lightgrey", "black"])
    fig, axs = plt.subplots(figsize=(5, 5))
    img = axs.imshow(object_gridd, cmap=cmap, origin="lower", extent=[0, 1, 0, 1])
    # plt.colorbar(img, ax=axs, label="Concentration", shrink=0.8)
    axs.set_title("2D Diffusion Simulation")

    def animate(frame):
        nonlocal  gridd, stencill, object_gridd, seedje
        seedje += frame
        gridd, object_gridd, stencill = solutions.perform_update_ADL(gridd, object_gridd, stencill, grid_indices, eta, seedje, sr_val)
        img.set_array(object_gridd)
        axs.set_title(f"DLA (η: {eta}) (Step: {frame:.3g})")

        if frame%100 == 0:
            print(f"finished first {frame} frames")
        return img,
    # Create animation
    animation = FuncAnimation(
        fig, animate, frames=2000, interval=interval, blit=True
    )
    animation.save(f"plots/2D_diffusion_{eta}.gif", fps=50, writer="ffmpeg")

    plt.close(fig)
    return animation
