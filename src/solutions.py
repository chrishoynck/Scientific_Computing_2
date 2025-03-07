import random as random

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
    delta = float("inf")

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
                west = grid[i, j - 1]  # if j > 0 else grid[i, N - 1]
                east = grid[i, j + 1]  # if j < N - 1 else grid[i, 0]

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
                west = grid[i, j - 1]  # if j > 0 else grid[i, N - 1]
                east = grid[i, j + 1]  # if j < N - 1 else grid[i, 0]

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

        # assert np.all(grid[0, :] == 1 ), "the top row is not 1 anymore"
        # grid[object_grid==1] = 0
        iter += 1

    return iter, grid


def perform_update_ADL(
    gridje, object_gridje, stenciltje, grid_indices, eta, seedje, SOR_pars
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

            print(f"starting experimentation for run {i} ")

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


def initialize_grid_with_cluster(N, cluster_row=0, cluster_col=15):
    """
    Initialize a square grid and set a cluster at a specified offset.

    Parameters:
        N (int): The size of the grid. The grid will be an N x N numpy array.
        cluster_row (int, optional): The base row index for the cluster's starting position. Defaults to 0.
        cluster_col (int, optional): The base column index for the cluster's starting position. Defaults to 15.

    Returns:
        tuple: A tuple containing:
            - grid (np.ndarray): An N x N grid with the cluster cell(s) marked as 1.
            - cluster_positions (list of tuple): A list of the relative cluster positions used to
              set the cluster on the grid.
    """

    grid = np.zeros((N, N))

    # Set the initial starting cluster as a single cell in the middle of the bottom row
    cluster_positions = [
        (0, 50),
    ]

    for dr, dc in cluster_positions:
        r = cluster_row + dr
        c = cluster_col + dc

        if 0 <= r < N and 0 <= c < N:
            grid[r, c] = 1
    return grid, cluster_positions


def monte_carlo_dla(N, p_join, cluster_length, animation=False):
    """
    Simulate a Diffusion-Limited Aggregation (DLA) process using a Monte Carlo approach.

    Parameters:
        N (int): The size of the square grid (N x N).
        p_join (float): The probability that a random walker will join the cluster upon contact.
        cluster_length (int): The target number of cells in the cluster at which to stop the simulation.
        animation (bool, optional): If True, intermediate grid states are stored for creating an animation.
                                    If False, only the final grid state is returned. Defaults to False.

    Returns:
        list: A list of numpy.ndarray objects representing grid states. When animation is True, this list
              contains multiple intermediate grid states; otherwise, it contains a single grid representing
              the final state of the simulation.
    """

    grid, cluster_positions = initialize_grid_with_cluster(N)
    all_grids = []
    current_walkers = []
    no_update = 0
    while len(cluster_positions) < cluster_length:
        new_walker = generating_random_walkers(cluster_positions, N, current_walkers)
        if new_walker is not None:
            current_walkers.append(new_walker)

        prev_len = len(cluster_positions)
        current_walkers, cluster_positions = moving_random_walkers(
            current_walkers, cluster_positions, N, p_join
        )
        if prev_len == len(cluster_positions):
            no_update += 1
        else:
            no_update = 0

        if p_join == 0.01 and no_update > 80000:
            print("No update for 50000 steps")
            break
        if p_join == 0.05 and no_update > 50000:
            print("No update for 40000 steps")
            break
        if p_join == 0.6 and no_update > 20000:
            print("No update for 10000 steps")
            break
        if p_join == 0.8 and no_update > 20000:
            print("No update for 5000 steps")
            break
        if p_join == 1 and no_update > 20000:
            print("No update for 1000 steps")
            break

        grid = np.zeros((N, N))

        for r, c in current_walkers:
            grid[r, c] = 2
        for r, c in cluster_positions:
            grid[r, c] = 1

        if (animation and len(cluster_positions) % 50 == 0) or (
            animation and len(cluster_positions) < 750
        ):
            all_grids.append(grid)

    if not animation:
        all_grids.append(grid)

    return all_grids


def generating_random_walkers(cluster_positions, N, current_walkers):
    """
    Create a random walker at the top of the grid if the selected position is free.

    Parameters:
        cluster_positions (list of tuple): A list of (row, column) tuples representing positions that are part of the cluster.
        N (int): The size of the grid (the grid is assumed to be N x N).
        current_walkers (list of tuple): A list of (row, column) tuples representing the positions of active walkers.

    Returns:
        tuple or None: The (row, column) position for the new walker if the chosen location is unoccupied;
                       otherwise, returns None.
    """
    col_position = np.random.randint(0, N - 1)
    if (N - 1, col_position) not in cluster_positions and (
        N,
        col_position,
    ) not in current_walkers:
        return (N - 1, col_position)

    return None


def adjacent_to_cluster(r, c, cluster_positions):
    """
    Check if a cell is adjacent to any cell in the cluster.

    Parameters:
        r (int): The row index of the cell.
        c (int): The column index of the cell.
        cluster_positions (iterable of tuple): An iterable containing (row, column) tuples that represent
                                                 the positions of cells in the cluster.

    Returns:
        bool: True if at least one of the four neighboring cells is in the cluster; False otherwise.
    """
    neighbors = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]

    # If any neighbor is in the cluster, we say that (r, c) is adjacent
    return any(neighbor in cluster_positions for neighbor in neighbors)


def moving_random_walkers(current_walkers, cluster_positions, N, p_join):
    """
    Move random walkers and update cluster positions based on a joining probability.

    Parameters:
        current_walkers (list of tuple): A list of (row, column) tuples representing the positions of active walkers.
        cluster_positions (list of tuple): A list of (row, column) tuples representing the positions of cells in the cluster.
        N (int): The size of the grid (i.e., the grid is N x N).
        p_join (float): The probability (between 0 and 1) that a walker adjacent to the cluster will join it.

    Returns:
        tuple: A tuple containing:
            - new_walkers (list of tuple): The updated list of walker positions after movement.
            - cluster_positions (list of tuple): The updated list of cluster positions, including any new additions from walkers joining.

    Notes:
        - If the next move places a walker outside the vertical grid boundaries, the walker is removed.
        - For horizontal moves, periodic boundary conditions are applied so that a walker exiting one side reappears on the opposite side.
    """

    new_walkers = []

    for r, c in current_walkers:
        north = (r + 1, c)
        south = (r - 1, c)
        east = (r, c + 1)
        west = (r, c - 1)
        directions = [north, south, east, west]

        next_r, next_c = random.choice(directions)
        # Removes walker if it goes outside the grid on the top or at the bottom
        if next_r < 0 or next_r >= N:
            new_col = np.random.randint(0, N)
            new_walker = (N - 1, new_col)
            # Optionally, check for conflicts with existing walkers or cluster positions.
            if new_walker not in cluster_positions and new_walker not in new_walkers:
                new_walkers.append(new_walker)
            continue

        # Periodic boundaries
        if next_c < 0:
            next_c = N - 1
        elif next_c >= N:
            next_c = 0

        if (
            (next_r, next_c) in current_walkers
            or (next_r, next_c) in new_walkers
            or (next_r, next_c) in cluster_positions
        ):
            new_walkers.append((r, c))
            continue

        if adjacent_to_cluster(next_r, next_c, cluster_positions):
            if random.random() < p_join:
                # print(f"Walker has joined the cluster at position {(next_r, next_c)}")
                cluster_positions.append((next_r, next_c))
            else:
                new_walkers.append((r, c))
        else:  # If the walker is not in the cluster or in the current_walkers list and the next move is free, move the walker to the position
            new_walkers.append((next_r, next_c))

    return new_walkers, cluster_positions

def initialize_grid_gray_scott(N, noise_level):
    """
    Initializes the U and V concentration fields for the Gray-Scott model.

    Parameters:
        N (int): Grid size (N x N).
        noise_level (float): Standard deviation of Gaussian noise to be added to the initial conditions.

    Returns:
        tuple: Two 2D numpy arrays representing the initial concentration fields U and V.
    """
    assert isinstance(N, int) and N > 0, "Grid size N must be a positive integer."
    
    u = np.ones((N, N)) * 0.5  # initialize u with 0.5 everywhere
    v = np.zeros((N, N))  # initialize v with 0.0

    # center region
    r = N // 20  # size of region
    center_x = N // 2  # center coordinates
    start, end = center_x - r, center_x + r  # bounds

    # small perturbation in the center of v
    v[start:end, start:end] = 0.25

    # noise for u in center
    noise_u = np.random.normal(0, noise_level, (2*r, 2*r)) 
    u[start:end, start:end] += noise_u

    # noise for v in center
    noise_v = np.random.normal(0, noise_level, (2*r, 2*r)) 
    v[start:end, start:end] += noise_v

    assert u.shape == (N, N) and v.shape == (N, N), "U and V grids must have shape (N, N)."
    return u, v

@njit(parallel=True)
def laplace(grid, dx):
    """
    Computes the discrete Laplacian of a given grid using finite differences with periodic boundary conditions, parallelized over 1D.

    Parameters:
        grid (numpy.ndarray): 2D array representing the grid.
        dx (float): Grid spacing.

    Returns:
        numpy.ndarray: The discrete Laplacian of the input grid.
    """
    assert isinstance(dx, (int, float)) and dx > 0, "Grid spacing dx must be a positive number."
    
    N = grid.shape[0]
    laplace_grid = np.zeros_like(grid)
    
    # parallel over rows
    for i in prange(N):  # row index
        for j in range(N):  # column index

            # periodic boundary conditions
            north = grid[i + 1, j] if i < N - 1 else grid[0, j]
            south = grid[i - 1, j] if i > 0 else grid[N - 1, j]
            east  = grid[i, j + 1] if j < N - 1 else grid[i, 0]
            west  = grid[i, j - 1] if j > 0 else grid[i, N - 1]

            laplace_grid[i, j] = (north + south + east + west - 4 * grid[i, j]) / (dx**2)
    
    assert laplace_grid.shape == grid.shape, "Laplacian output must have the same shape as input grid."
    return laplace_grid

def update_gray_scott(u, v, num_steps, N, dt, dx, Du, Dv, f, k):
    """
    Simulates the Gray-Scott reaction-diffusion process and returns the final concentration fields.
    
    Parameters:
        u (numpy.ndarray): Initial concentration of U.
        v (numpy.ndarray): Initial concentration of V.
        num_steps (int): Number of simulation time steps.
        N (int): Grid size.
        dt (float): Time step size.
        dx (float): Grid spacing.
        Du (float): Diffusion coefficient for U.
        Dv (float): Diffusion coefficient for V.
        f (float): Feed rate.
        k (float): Kill rate.
    
    Returns:
        tuple: Final concentration fields U and V as 2D numpy arrays.
    """
    for i in range(num_steps):
        Lu = laplace(u, dx)
        Lv = laplace(v, dx)

        # smallest_value = np.finfo(np.float64).tiny  # Smallest positive float
        # v = np.maximum(v, smallest_value)
        # u = np.maximum(u, smallest_value)

        uvv = u * np.square(v)

        # uvv = u * v**2
        # du_dt = Du * Lu - uvv + f * (1 - u) + noise_level * np.random.randn(N, N)
        # dv_dt = Dv * Lv + uvv - (f + k) * v + noise_level * np.random.randn(N, N)
        du_dt = Du * Lu - uvv + f * (1 - u)
        dv_dt = Dv * Lv + uvv - (f + k) * v
        
        u += du_dt * dt
        v += dv_dt * dt
    
    assert u.shape == (N, N) and v.shape == (N, N), "Final output must have shape (N, N)."

    return u, v

def run_simulation_gray_scott(N, num_steps, dt, dx, Du, Dv, f, k, noise_level):
    """
    Runs the Gray-Scott reaction-diffusion simulation.
    
    Parameters:
        N (int): Grid size.
        num_steps (int): Number of time steps.
        dt (float): Time step size.
        dx (float): Grid spacing.
        Du (float): Diffusion coefficient for U.
        Dv (float): Diffusion coefficient for V.
        f (float): Feed rate.
        k (float): Kill rate.
        noise_level (float): Amplitude of noise added to the system.
    
    Returns:
        tuple: Final concentration fields U and V.
    """
    u, v = initialize_grid_gray_scott(N, noise_level)
    u_final, v_final = update_gray_scott(u, v, num_steps, N, dt, dx, Du, Dv, f, k)

    assert u_final.shape == (N, N) and v_final.shape == (N, N), "Final output must have shape (N, N)."

    return u_final, v_final