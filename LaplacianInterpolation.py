# % Class to perform 3D Laplacian Interpolation
# % EECE 8395: Engineering for Surgery
# % Fall 2023
# % Author: Prof. Jack Noble; jack.noble@vanderbilt.edu

import numpy as np
from scipy import sparse
from BICG import *

#w = x + r*y + r*c*z
def laplacianInterpolation(r,c,d,w,f):
    X,Y,Z = np.meshgrid(np.arange(0,r), np.arange(0,c), np.arange(0,d), indexing='ij')
    X = X.ravel(order='F').astype(np.longlong)
    Y = Y.ravel(order='F').astype(np.longlong)
    Z = Z.ravel(order='F').astype(np.longlong)

    rc = r * c
    rd = r * d
    cd = c * d
    N = np.array(rc*d, dtype=np.longlong) # number of nodes
    M = np.array(2*(rc + rd + cd), dtype=np.longlong) # number of 'ghost' nodes
    D = np.size(w) # number of Dirichlet nodes
    P = 7*(N-D) + D + 2*M # number of nonzero entries in sparse matrix

    I = np.zeros(P, dtype=np.intc)
    J = np.zeros(P, dtype=np.intc)
    V = np.zeros(P, dtype=np.double)


    rows1 = np.repeat(np.arange(0,N)[:,np.newaxis],7,axis=1)
    cols1 = rows1 + np.repeat(np.array([0, -1, 1, -r, r, -rc, rc])[np.newaxis,:], N, axis=0)


    rows2 = np.repeat(np.arange(N, N+M)[:,np.newaxis], 2, axis=1)
    cols2 = np.copy(rows2)
    V2 = np.ones((M,2))
    V2[:,1] = -1

    # X==0
    bndind = np.arange(0, N, r)
    cols1[bndind,1] = N + Y[bndind] + Z[bndind]*c
    cols2[Y[bndind] + Z[bndind]*c,1] = bndind

    # X==r-1
    bndind = r-1 + np.arange(0, N, r)
    cols1[bndind,2] = N + cd + Y[bndind] + Z[bndind]*c
    cols2[cd + Y[bndind] + Z[bndind]*c,1] = bndind

    # Y==0
    bndind = (np.repeat(np.arange(0, r)[np.newaxis,  :], d, axis=0) +
              np.repeat(np.arange(0,N,rc)[:,np.newaxis], r, axis=1)).ravel()
    cols1[bndind,3] = N + 2*cd + X[bndind] + Z[bndind]*r
    cols2[2*cd + X[bndind] + Z[bndind]*r,1] = bndind

    # Y==c-1
    bndind += r*(c-1)
    cols1[bndind,4] = N + 2*cd + rd + X[bndind] + Z[bndind]*r
    cols2[2*cd + rd + X[bndind] + Z[bndind]*r,1] = bndind

    # Z==0
    bndind = np.arange(0,rc)
    cols1[bndind, 5] = N + 2 * (cd + rd) + X[bndind] + Y[bndind] * r
    cols2[2 * (cd + rd) + X[bndind] + Y[bndind] * r + Z[bndind] * r, 1] = bndind

    # Z==d-1
    bndind += rc*(d-1)
    cols1[bndind, 6] = N + 2 * (cd + rd) + rc + X[bndind] + Y[bndind] * r
    cols2[2 * (cd + rd) + rc + X[bndind] + Y[bndind] * r, 1] = bndind

    # For Laplacian, first column of values should be equal to negative sum of all others
    # If doing non-homogenous electrostatics, these coefficients need to be redefined using the node conductivities
    # Hint: use a (N+M) element helper array 'cndty_withghost' that contains the regular node
    #   conductivities in entries [0,N-1] and progressively defines the ghost node conductivities in entries
    #   [N,N+M-1] while the column indices are getting updated for the Neumann conditions, above. The
    #   conductivity for a ghost node should be equal to the conductivity of its corresponding regular
    #   node. If done correctly the conductivity for the neighbors of node i can be accessed to help define values for V1 by
    #   doing cndty_withghost[cols1[i,1:]]
    V1 = np.ones((N, 7))
    V1[:, 0] = -np.sum(V1[:, 1:], axis=1)

    # Use mask for Dirichlet conditions
    msk = np.ones(np.shape(rows1), dtype=bool)
    msk[w, 1:] = 0
    V1[w,0] = 1
    I[0:(P-2*M)] = rows1[msk].ravel()
    J[0:(P-2*M)] = cols1[msk].ravel()
    V[0:(P-2*M)] = V1[msk].ravel()
    I[P-2*M:] = rows2.ravel()
    J[P-2*M:] = cols2.ravel()
    V[P-2*M:] = V2.ravel()

    b = np.zeros(N + M, dtype=np.double)
    b[w] = f

    # can also use sparse.linalg.spsolve from scipy.sparse instead of bicg
    # A = sparse.coo_matrix((V, (I,J)), shape=(N+M, N+M))
    # x = sparse.linalg.spsolve(A.tocsc(),b)

    bc = bicg()
    x = bc.Solve(N+M, N+M, I, J, V, b, maxiter=1000)
    phi = np.reshape(x[0:N], [r,c,d], order='F')

    return phi