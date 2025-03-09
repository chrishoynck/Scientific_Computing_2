import numpy as np


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
    midpoint = int(N / 2)
    # Set the initial starting cluster as a single cell in the middle of the bottom row
    cluster_positions = [
        (0, midpoint),
    ]

    for dr, dc in cluster_positions:
        r = cluster_row + dr
        c = cluster_col + dc

        if 0 <= r < N and 0 <= c < N:
            grid[r, c] = 1
    return grid, cluster_positions


def monte_carlo_dla(N, p_join, cluster_length, seedje, animation=False):
    """
    Simulate a Diffusion-Limited Aggregation (DLA) process using a Monte Carlo approach.

    Parameters:
        N (int): The size of the square grid (N x N).
        p_join (float): The probability that a random walker will join the cluster upon contact.
        cluster_length (int): The target number of cells in the cluster at which to stop the simulation.
        seedje (int): Random seed for reproducibility.
        animation (bool, optional): If True, intermediate grid states are stored for creating an animation.
                                    If False, only the final grid state is returned. Defaults to False.

    Returns:
        list: A list of numpy.ndarray objects representing grid states. When animation is True, this list
              contains multiple intermediate grid states; otherwise, it contains a single grid representing
              the final state of the simulation.
    """

    # initialize grid with placement of seed of the cluster
    grid, cluster_positions = initialize_grid_with_cluster(N)
    all_grids = []
    current_walkers = []
    no_update = 0
    while len(cluster_positions) < cluster_length:
        seedje += 1
        # place walkers on the grid, starting one new walker at the top every time-step
        new_walker = generating_random_walkers(
            cluster_positions, N, current_walkers, seedje
        )
        if new_walker is not None:
            current_walkers.append(new_walker)

        prev_len = len(cluster_positions)
        current_walkers, cluster_positions = moving_random_walkers(
            current_walkers, cluster_positions, N, p_join, seedje
        )
        # for check if structure is still developing
        if prev_len == len(cluster_positions):
            no_update += 1
        else:
            no_update = 0

        # check if development is stagnated
        if p_join == 0.01 and no_update > 80000:
            print("No update for 80000 steps")
            break
        elif p_join == 0.05 and no_update > 50000:
            print("No update for 50000 steps")
            break
        elif p_join == 0.6 and no_update > 20000:
            print("No update for 20000 steps")
            break
        elif p_join == 0.8 and no_update > 20000:
            print("No update for 20000 steps")
            break
        else:
            if no_update > 20000:
                print("No update for 20000 steps")
                break

        grid = np.zeros((N, N))

        # set values right, non occupied 0, cluster 1, random walker 2
        for r, c in current_walkers:
            grid[r, c] = 2
        for r, c in cluster_positions:
            grid[r, c] = 1

        # save grid
        if (animation and len(cluster_positions) % 300 == 0) or (
            animation and len(cluster_positions) < 750
        ):
            all_grids.append(grid)

    if not animation:
        all_grids.append(grid)

    return all_grids


def generating_random_walkers(cluster_positions, N, current_walkers, seedje):
    """
    Create a random walker at the top of the grid if the selected position is free.

    Parameters:
        cluster_positions (list of tuple): A list of (row, column) tuples representing positions that are part of the cluster.
        N (int): The size of the grid (the grid is assumed to be N x N).
        current_walkers (list of tuple): A list of (row, column) tuples representing the positions of active walkers.
        seedje (int): Random seed for reproducibility.

    Returns:
        tuple or None: The (row, column) position for the new walker if the chosen location is unoccupied;
                       otherwise, returns None.
    """

    np.random.seed(seedje)

    # generate random int within domain
    col_position = np.random.randint(0, N - 1)

    # check if occupied
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


def moving_random_walkers(current_walkers, cluster_positions, N, p_join, seedje):
    """
    Move random walkers and update cluster positions based on a joining probability.

    Parameters:
        current_walkers (list of tuple): A list of (row, column) tuples representing the positions of active walkers.
        cluster_positions (list of tuple): A list of (row, column) tuples representing the positions of cells in the cluster.
        N (int): The size of the grid (i.e., the grid is N x N).
        p_join (float): The probability (between 0 and 1) that a walker adjacent to the cluster will join it.
        seedje (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing:
            - new_walkers (list of tuple): The updated list of walker positions after movement.
            - cluster_positions (list of tuple): The updated list of cluster positions, including any new additions from walkers joining.

    Notes:
        - If the next move places a walker outside the vertical grid boundaries, the walker is removed.
        - For horizontal moves, periodic boundary conditions are applied so that a walker exiting one side reappears on the opposite side.
    """

    np.random.seed(seedje)

    new_walkers = []

    for r, c in current_walkers:
        north = (r + 1, c)
        south = (r - 1, c)
        east = (r, c + 1)
        west = (r, c - 1)
        directions = [north, south, east, west]

        # determine next direction
        next_r, next_c = directions[np.random.randint(len(directions))]

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

        # place walker wants to go to already occupied, it stays at its current place
        if (
            (next_r, next_c) in current_walkers
            or (next_r, next_c) in new_walkers
            or (next_r, next_c) in cluster_positions
        ):
            new_walkers.append((r, c))
            continue

        if adjacent_to_cluster(next_r, next_c, cluster_positions):
            # see if walker joins cluster, using the probability to react parameter
            if np.random.rand() < p_join:
                cluster_positions.append((next_r, next_c))
            else:
                new_walkers.append((r, c))
        else:  # If the walker is not in the cluster or in the current_walkers list and the next move is free, move the walker to the position
            new_walkers.append((next_r, next_c))

    return new_walkers, cluster_positions
