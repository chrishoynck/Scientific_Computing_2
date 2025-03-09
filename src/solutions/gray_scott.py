import numpy as np
from numba import njit, prange

def initialize_grid_gray_scott(N, noise_level, seedje):
    """
    Initializes the U and V concentration fields for the Gray-Scott model.

    Parameters:
        N (int): Grid size (N x N).
        noise_level (float): Standard deviation of Gaussian noise to be added to the initial conditions.
        seedje (int): Random seed for reproducibility.

    Returns:
        tuple: Two 2D numpy arrays representing the initial concentration fields U and V.
    """
    assert isinstance(N, int) and N > 0, "Grid size N must be a positive integer."
    
    np.random.seed(seedje)

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

        uvv = u * np.square(v)
        du_dt = Du * Lu - uvv + f * (1 - u)
        dv_dt = Dv * Lv + uvv - (f + k) * v
        
        u += du_dt * dt
        v += dv_dt * dt
    
    assert u.shape == (N, N) and v.shape == (N, N), "Final output must have shape (N, N)."

    return u, v

def run_simulation_gray_scott(N, num_steps, dt, dx, Du, Dv, f, k, noise_level, seedje):
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
    u, v = initialize_grid_gray_scott(N, noise_level, seedje)
    u_final, v_final = update_gray_scott(u, v, num_steps, N, dt, dx, Du, Dv, f, k)

    assert u_final.shape == (N, N) and v_final.shape == (N, N), "Final output must have shape (N, N)."

    return u_final, v_final