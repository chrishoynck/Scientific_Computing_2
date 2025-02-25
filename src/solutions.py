import numpy as np

def initialize_grid(N):
    """
    Generates a grid with the specified dimensions and initializes the boundaries.
    Parameters:
        N (int): Grid size.
    """
    assert N > 1, "Grid size must be bigger than 1x1"
    grid = np.zeros((N, N))

    return grid

def sequential_SOR(grid,tol, max_iters, omega, object_grid=None):
    """
    Solves using the Successive Over Relaxtion (SOR) iteration method.

    The update equation is:
        c_{i,j}^{k+1} = (omega/4) * (c_{i+1,j}^{k} + c_{i,j+1}^{k} + c_{i,j+1}^{k} + (1 - omega) c_{i,j}^{k})

    Parameters:
        N (int): Grid size.
        tol (float): Convergence tolerance.
        max_iters (int): Maximum number of iterations.
        omega (float): Relaxation factor.

    Returns:
        int: Number of iterations required to reach convergence.
        numpy.ndarray: Final grid after iterations.
    """
    N= len(grid)
    assert N > 1, (
        f"bord is {N}x{N}, but needs to be at least 2*2 for this diffusion implementation"
    )

    # grid initialisation
    # c = initialize_grid(N)

    iter = 0
    delta = float("inf")

    # while not converged
    while delta > tol and iter < max_iters:
        delta = 0

        # loop over all cells in the grid (except for y = 0, y=N)
        for i in range(1, N - 1):
            for j in range(N):
                # retrieve all necessary values (also regarding wrap-around)
                south = grid[i - 1, j] if i > 0 else 0
                north = grid[i + 1, j] if i < N - 1 else 1
                west = grid[i, j - 1] if j > 0 else grid[i, N - 1]
                east = grid[i, j + 1] if j < N - 1 else grid[i, 0]

                # SOR update equation
                c_next = (omega / 4) * (west + east + south + north) + (1 - omega) * grid[
                    i, j
                ]

                # check for convergence
                delta = max(delta, abs(c_next - grid[i, j]))
                grid[i, j] = c_next

        iter += 1

    return iter, grid

def prob(cell, cell_candidates):
    return 