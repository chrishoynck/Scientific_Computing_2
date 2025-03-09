import os
import pickle
import time

import numpy as np
from numba import njit, prange
from scipy.ndimage import binary_dilation


def place_objects(N, size_object=1):
    """
    places square objects on an NxN grid, at the bottom (seed of DLA)

    Parameters:
        N (int): Grid size (N × N).
        num_object (int): Number of objects to place.
        seed (int, optional): Random seed for reproducibility (default=31).
        size_object (int, optional): Side length of each square object (default=4).

    Returns:
        numpy.ndarray: NxN grid with placed objects, where occupied cells are marked as 1.
    """

    object_grid = np.zeros((N, N))
    y = int(N / 2 - size_object / 2)

    points = [
        (N - 1 - k, y + j) for j in range(size_object) for k in range(size_object)
    ]

    # set value of indexes that object occupies 1
    object_grid[tuple(zip(*points))] = 1

    return object_grid


def generate_stencil(object_grid):
    """
    Generates a stencil indicating valid growth candidates in a diffusion-limited aggregation model.

    The stencil marks grid points adjacent to existing objects using a Neumann (4-neighbor) neighborhood.
    Original object positions remain unchanged.

    Parameters:
        object_grid (numpy.ndarray): A 2D grid where object locations are marked as 1.

    Returns:
        numpy.ndarray: A 2D stencil grid with candidate growth sites marked as 1.
    """
    stencil = np.copy(object_grid)
    # Define a strict Neumann (4-neighbor) structuring element
    neighborhood = np.array(
        [
            [0, 1, 0],  # Only North, South, West, East
            [1, 0, 1],
            [0, 1, 0],
        ]
    )

    # Apply binary dilation to get the stencil
    stencil = binary_dilation(object_grid, structure=neighborhood).astype(int)

    # Ensure original object points remain unchanged
    stencil[object_grid == 1] = 0  # Exclude original object points
    return stencil


def save_grid_to_file(grid, filename="data/grid_output.txt"):
    np.savetxt(filename, grid, fmt="%.6f")  # Save with 6 decimal places for precision
    print(f"Grid saved to {filename}")


def load_grid_from_file(filename="data/grid_output.txt"):
    return np.loadtxt(filename)  # Loads the 2D array back


def write_concentration_DLA(gridjes, filename="gridjes.pkl"):
    """
    Saves concentration grids to a pickle file.

    Parameters:
    gridjes (dict): Dictionary mapping each η value to a tuple (grid, object_grid), where:
        - grid (numpy.ndarray): 2D array representing concentration values.
        - object_grid (numpy.ndarray): 2D array indicating object placements.
    filename (str): Name of the pickle file to save. Default is "gridjes.pkl".
    """
    complete_dir = "data/DLA_concentrations"
    os.makedirs(
        complete_dir, exist_ok=True
    )  # This will create the folder if it doesn't exist.

    file_path = os.path.join(complete_dir, filename)
    assert os.path.exists(complete_dir), f"{complete_dir} is not a valid directory"
    with open(file_path, "wb") as f:
        pickle.dump(gridjes, f)
    print(f"grids saved to {complete_dir}")


def read_in_concentration_DLA(filename="gridjes.pkl"):
    """
    Loads concentration grids from a pickle file.

    Parameters:
    filename (str): Name of the pickle file to load. Default is "gridjes.pkl".

    Returns:
    dict: Dictionary mapping each η value to a tuple (grid, object_grid), where:
        - grid (numpy.ndarray): 2D array representing concentration values.
        - object_grid (numpy.ndarray): 2D array indicating object placements.
    """
    complete_dir = f"data/DLA_concentrations/{filename}"
    assert os.path.exists(complete_dir), f"{complete_dir} is not a valid directory"
    with open(complete_dir, "rb") as f:
        gridjes_loaded = pickle.load(f)
    return gridjes_loaded


def initialize_grid(N, object_grid):
    """
    Generates a grid with the specified dimensions and initializes the boundaries.
    Parameters:
        N (int): Grid size.
        object_grid (np.array): 2D matrix with 1s one places where the object is placed (0s everywhere else)
    """

    # assert parameters are suitable for this implementation
    assert N > 1, "Grid size must be bigger than 1x1"
    assert object_grid.shape == (N, N), (
        f"object_grid must have the same dimensions as {N, N}"
    )

    grid = np.zeros((N, N))

    grid[0, :] = 1  # bottom boundary
    grid[N - 1, :] = 0  # top boundary

    # objects are sinks
    grid[object_grid == 1] = 0
    return grid


def empty_object_places(grid, stenciltje, object_grid, eta):
    """
    Computes the growth probabilities for candidate sites in a diffusion-limited aggregation model.

    The function extracts valid growth sites from the diffusion grid, normalizes their values,
    and applies an exponent `nu` to control growth bias.

    Parameters:
        grid (numpy.ndarray): The 2D diffusion grid representing concentration values.
        stenciltje (numpy.ndarray): A stencil marking valid growth candidate locations.
        object_grid (numpy.ndarray): The 2D grid indicating object placements.
        nu (float): Growth parameter that modifies probability distribution.

    Returns:
        numpy.ndarray: A normalized probability distribution over valid growth sites.
    """
    assert object_grid.shape == grid.shape, (
        "object_grid must have the same dimensions as diffusion grid"
    )

    # consider only potential new cells
    emptied_grid = np.copy(grid)
    emptied_grid[stenciltje == 0] = 0

    # numerical errors can cause a value to slightly drop below zero -> set to zero
    if np.any(emptied_grid < 0):
        emptied_grid[emptied_grid < 0] = 0

    # apply eta parameter: how strongly the concentration is involved in probability
    emptied_grid = np.power(emptied_grid, eta)
    total_sum = emptied_grid.sum()

    # if sum is 0, no probabilities are assigned, so no object or diffusion source is present
    assert total_sum > 0, (
        "Initialize object or source, The Advection Diffusion does not work on an empty grid"
    )
    emptied_grid /= total_sum

    return emptied_grid.flatten()


@njit
def sequential_SOR(grid, tol, max_iters, omega, object_grid=None):
    """
    Solves using the Successive Over Relaxtion (SOR) iteration method.

    The update equation is:
        c_{i,j}^{k+1} = (omega/4) * (c_{i+1,j}^{k} + c_{i,j+1}^{k} + c_{i,j+1}^{k} + (1 - omega) c_{i,j}^{k})

    Parameters:
        grid (np.array): Grid (2D matrix).
        tol (float): Convergence tolerance.
        max_iters (int): Maximum number of iterations.
        omega (float): Relaxation factor.

    Returns:
        int: Number of iterations required to reach convergence.
        numpy.ndarray: Final grid after iterations.
    """
    N = len(grid)
    assert N > 1, (
        f"bord is {N}x{N}, but needs to be at least 2*2 for this diffusion implementation"
    )

    # grid initialisation
    # c = initialize_grid(N)

    iter = 0
    delta = np.inf

    # while not converged
    while delta > tol and iter < max_iters:
        delta = 0

        # loop over all cells in the grid (except for y = 0, y=N)
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                if object_grid is not None and object_grid[(i, j)]:
                    c_next = 0
                    continue
                # retrieve all necessary values (also regarding wrap-around)
                south = grid[i - 1, j] if i > 0 else 1
                north = grid[i + 1, j] if i < N - 1 else 0
                west = grid[i, j - 1]
                east = grid[i, j + 1]

                # SOR update equation
                c_next = (omega / 4) * (west + east + south + north) + (
                    1 - omega
                ) * grid[i, j]

                # check for convergence
                delta = max(delta, abs(c_next - grid[i, j]))
                grid[i, j] = c_next

        # borders, derivative is 0 at the borders
        grid[:, N - 1] = grid[:, N - 2]
        grid[:, 0] = grid[:, 1]
        # grid[object_grid==1] = 0
        iter += 1

    return iter, grid


@njit(parallel=True)
def parallel_SOR(grid, tol, max_iters, omega, object_grid=None):
    """
    Solves using the Successive Over Relaxtion (SOR) iteration method. Uses red-black block coloring scheme for parallelization

    The update equation is:
        c_{i,j}^{k+1} = (omega/4) * (c_{i+1,j}^{k} + c_{i,j+1}^{k} + c_{i,j+1}^{k} + (1 - omega) c_{i,j}^{k})

    Parameters:
        grid (np.array): Grid (2D matrix).
        tol (float): Convergence tolerance.
        max_iters (int): Maximum number of iterations.
        omega (float): Relaxation factor.

    Returns:
        int: Number of iterations required to reach convergence.
        numpy.ndarray: Final grid after iterations.
    """
    N = len(grid)
    assert N > 1, (
        f"bord is {N}x{N}, but needs to be at least 2*2 for this diffusion implementation"
    )

    iter = 0
    delta = np.inf

    # while not converged
    while delta > tol and iter < max_iters:
        delta = 0

        # loop over all cells in the grid (except for y = 0, y=N) alternating between uneven and even row indexes
        for i in prange(1, N - 1):
            if i % 2 == 0:
                start = 1
                end = N - 2
            else:
                start = 2
                end = N - 1
            for j in range(start, end, 2):
                if object_grid is not None and object_grid[(i, j)]:
                    c_next = 0
                    continue
                # retrieve all necessary values (also regarding wrap-around)
                south = grid[i - 1, j] if i > 0 else 1
                north = grid[i + 1, j] if i < N - 1 else 0
                west = grid[i, j - 1]
                east = grid[i, j + 1]

                # SOR update equation
                c_next = (omega / 4) * (west + east + south + north) + (
                    1 - omega
                ) * grid[i, j]

                # check for convergence
                delta = max(delta, abs(c_next - grid[i, j]))
                grid[i, j] = c_next

        # loop over all cells in the grid, alternating between uneven and even row indexes (except for y = 0, y=N)
        for i in prange(1, N - 1):
            if i % 2 == 0:
                start = 2
                end = N - 1
            else:
                start = 1
                end = N - 2
            for j in range(start, end, 2):
                if object_grid is not None and object_grid[(i, j)]:
                    c_next = 0
                    continue
                # retrieve all necessary values (also regarding wrap-around)
                south = grid[i - 1, j] if i > 1 else 1
                north = grid[i + 1, j] if i < N - 2 else 0
                west = grid[i, j - 1]
                east = grid[i, j + 1]

                # SOR update equation
                c_next = (omega / 4) * (west + east + south + north) + (
                    1 - omega
                ) * grid[i, j]

                # check for convergence
                delta = max(delta, abs(c_next - grid[i, j]))
                grid[i, j] = c_next

            # borders, derivative is 0 at the borders
            grid[i, N - 1] = grid[i, N - 2]
            grid[i, 0] = grid[i, 1]
        iter += 1

    return iter, grid


def perform_update_ADL(
    gridje,
    object_gridje,
    stenciltje,
    grid_indices,
    eta,
    seedje,
    SOR_pars,
    sequential=False,
):
    """
    Performs a single update step in the Aggregation Diffusion Limited (ADL) process.
    This function updates the diffusion grid using Successive Over-Relaxation (SOR).

    Parameters:
        gridje (numpy.ndarray): The 2D diffusion grid representing the current state.
        object_gridje (numpy.ndarray): The 2D grid indicating object placements.
        stenciltje (numpy.ndarray): The 2D stencil grid used to determine growth candidates.
        eta (float): Growth parameter that affects probability calculations.
        SOR_pars (tuple): Parameters for the SOR method, containing:
            - tol (float): Convergence tolerance for SOR.
            - maxiters (int): Maximum number of SOR iterations.
            - omega (float): Relaxation factor for SOR.

    Returns:
        tuple:
            - gridje (numpy.ndarray): Updated 2D diffusion grid.
            - object_gridje (numpy.ndarray): Updated object grid with a new placement.
            - stenciltje (numpy.ndarray): Updated stencil grid after placement.
    """

    # set seed for reproducability
    np.random.seed(seedje)

    # extract parameters for sor
    (tol, maxiters, omega) = SOR_pars

    # do SOR convergence for this grid
    if sequential:
        iters, gridje = sequential_SOR(gridje, tol, maxiters, omega, object_gridje)
    else:
        iters, gridje = parallel_SOR(gridje, tol, maxiters, omega, object_gridje)

    assert iters < maxiters, f"No convergence for SOR, omega: {omega}"

    # create stencil around object, which are the potential cells joining the object
    stenciltje = generate_stencil(object_gridje)

    # generate probabilities associated with each object
    probs = empty_object_places(gridje, stenciltje, object_gridje, eta)
    selected_index = np.random.choice(grid_indices, p=probs)
    new_index = np.unravel_index(selected_index, gridje.shape)

    # set the object grid of this new joined cell to 1
    object_gridje[new_index] = 1
    gridje[new_index] = 0

    return gridje, object_gridje, stenciltje, iters


def optimize_omega_DLA(
    itertjes, etas, seedje, omegas, grid, object_grid, tol, maxiters, grid_indices
):
    """
    Optimizes omega parameter for a 2D DLA process.

    Runs multiple experiments (10) to determine the optimal omega value for the Successive Over-Relaxation (SOR)
    solver by evaluating the average number of iterations required for convergence over a given number of steps.

    Parameters:
        itertjes (int): Number of iterations for the DLA process per run.
        seedje (int): Seed value for random number generation to ensure reproducibility.
        omegas (list of float): List of omega values to be tested.
        grid (numpy.ndarray): Initial 2D grid representing concentration values.
        object_grid (numpy.ndarray): 2D grid indicating object placements.
        tol (float): Convergence tolerance for the SOR solver.
        maxiters (int): Maximum number of iterations allowed for SOR convergence.
        grid_indices (numpy.ndarray): Indices used for random selection in the DLA process.

    Returns:
        eta_list (dict): Dictionary mapping each η value to a list of recorded SOR iterations for each omega.
        grid_list (dict): Dictionary mapping each η value to the final state of the grid and object grid.
    """

    eta_list = dict()
    grid_list = dict()
    eta_iters = dict()

    # loop over all etas
    for eta in etas:
        print(f"Starting finding optimal omega for η: {eta}")
        all_omega_iters = []
        all_mean_iters = []

        # perfrom this 10 times (stochastic process)
        for i in range(10):
            omegas_iters = dict()
            all_iters_dict = dict()
            seedje += i

            # iterate over all omega values
            for j, omega in enumerate(omegas):
                # use empty grid (initialized)
                iter_grid = np.copy(grid)
                object_grid_iter = np.copy(object_grid)
                Sr_pars = (tol, maxiters, omega)

                # generate initial potential new cells for the object
                stencil_iter = generate_stencil(object_grid_iter)
                seedje += j

                total_sor_iters = 0
                all_itertjes = []
                for i in range(itertjes):
                    seedje += i

                    # update grid
                    iter_grid, object_grid_iter, stencil_iter, sor_iters = (
                        perform_update_ADL(
                            iter_grid,
                            object_grid_iter,
                            stencil_iter,
                            grid_indices,
                            eta,
                            seedje,
                            Sr_pars,
                        )
                    )
                    total_sor_iters += sor_iters
                    all_itertjes.append(sor_iters)

                # compute average number of iterations until convergence
                total_sor_iters /= itertjes

                # save the number of iterations until conergence
                omegas_iters[omega] = total_sor_iters
                all_iters_dict[omega] = all_itertjes

            # save all the iterations for every run
            all_omega_iters.append(omegas_iters)
            all_mean_iters.append(all_iters_dict)

        # save all iterations and one configuration for all the eta values
        grid_list[eta] = (iter_grid, object_grid_iter)
        eta_list[eta] = all_omega_iters
        eta_iters[eta] = all_mean_iters
    return eta_list, grid_list, eta_iters


def speedup_par_seq(grid, object_grid, sr_val, grid_indices, clus_size, etatje=1):
    """
    Compares execution time of parallel and sequential implementations of a 2D update process.

    Parameters:
        grid (numpy.ndarray): Initial 2D grid configuration.
        object_grid (numpy.ndarray): 2D grid indicating object placements.
        sr_val (float): Relaxation parameter for SOR updates.
        grid_indices (numpy.ndarray or list): Indices used for random selection in updates.
        clus_size (int): Number of update iterations.
        etatje (float, optional): Control parameter for updates (default is 1).

    Returns:
        parallel_time (float): Execution time of the parallel implementation.
        sequential_time (float): Execution time of the sequential implementation.
    """
    parallel_grid = np.copy(grid)
    stencil_parallel = generate_stencil(parallel_grid)
    object_grid_par = np.copy(object_grid)

    stencil_seq = np.copy(stencil_parallel)
    sequential_grid = np.copy(grid)
    object_grid_seq = np.copy(object_grid)

    seedje = 22
    total_iters = 0

    start_parallel = time.time()
    for _ in range(clus_size):
        seedje += 1
        parallel_grid, object_grid_par, stencil_parallel, itertjes = perform_update_ADL(
            parallel_grid,
            object_grid_par,
            stencil_parallel,
            grid_indices,
            etatje,
            seedje,
            sr_val,
        )
        total_iters += itertjes
    end_parallel = time.time()
    print(
        f"The total number of SOR iterations for the parallel implementation is {total_iters}"
    )

    total_iters = 0
    start_sequential = time.time()
    seedje = 22
    for _ in range(clus_size):
        seedje += 1
        sequential_grid, object_grid_seq, stencil_seq, _ = perform_update_ADL(
            sequential_grid,
            object_grid_seq,
            stencil_seq,
            grid_indices,
            etatje,
            seedje,
            sr_val,
            True,
        )
        total_iters += itertjes
    end_sequential = time.time()
    print(
        f"The total number of SOR iterations for the parallel implementation is {total_iters}"
    )

    parallel_time = end_parallel - start_parallel
    sequential_time = end_sequential - start_sequential

    return (
        parallel_time,
        sequential_time,
        (parallel_grid, object_grid_par),
        (sequential_grid, object_grid_seq),
    )
