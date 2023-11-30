import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import ujson as json

def electroStaticFDM(cndty, nds, volts):
    r, c, d = cndty.shape  # dimensions of the grid
    grid_size = r * c * d  # total number of points in the grid

    # Initialize the potential array
    phi = np.zeros(grid_size)

    # Helper function to convert 3D indices to 1D index
    def idx_3d_to_1d(i, j, k):
        return i * c * d + j * d + k

    # Precompute 1D indices for the whole grid
    idx_3d = np.array(np.meshgrid(np.arange(r), np.arange(c), np.arange(d), indexing='ij'))
    idx_1d = idx_3d_to_1d(idx_3d[0], idx_3d[1], idx_3d[2])

    # Vectorized construction of finite difference coefficients
    # We use np.roll to shift the conductivity matrix and compute coefficients
    main_diag = -2 * (np.roll(cndty, 1, axis=0) + cndty + np.roll(cndty, 1, axis=1) + np.roll(cndty, 1, axis=2))
    off_diag_x = np.roll(cndty, -1, axis=0)  # shifted along x-axis
    off_diag_y = np.roll(cndty, -1, axis=1)  # shifted along y-axis
    off_diag_z = np.roll(cndty, -1, axis=2)  # shifted along z-axis

    # Flatten the coefficient arrays for sparse matrix construction
    main_diag_flat = main_diag.flatten()
    off_diag_x_flat = off_diag_x.flatten()
    off_diag_y_flat = off_diag_y.flatten()
    off_diag_z_flat = off_diag_z.flatten()

    # Construct the sparse matrix A
    A = sp.diags([main_diag_flat, off_diag_x_flat, off_diag_x_flat, off_diag_y_flat, off_diag_y_flat, off_diag_z_flat, off_diag_z_flat],
                 [0, r * d, -r * d, d, -d, 1, -1], shape=(grid_size, grid_size), format='csr')

    # Set known voltages
    B = np.zeros(grid_size)
    for idx, nd in enumerate(nds):
        phi_flat_idx = idx_3d_to_1d(*nd)
        B[phi_flat_idx] = volts[idx]
        A[phi_flat_idx, :] = 0
        A[phi_flat_idx, phi_flat_idx] = 1

    # Solve the linear system
    phi = spla.spsolve(A, B)

    # Reshape the solution to the original [r, c, d] format
    phi = phi.reshape((r, c, d))

    return phi

if __name__ == "__main__": 

    with open('Project6.json', 'r') as file:
        project_data = json.load(file)