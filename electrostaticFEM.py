import numpy as np
import skfem
from BICG import bicg
import time


# create synthetic conductivity map and try FEM
r = 31
c = 41
d = 51

@skfem.BilinearForm
def form(u, v, w):
    return (1/(w['h'])) * (np.abs(u.grad[0]*u.grad[0] + 
                                    u.grad[1]*u.grad[1]) * 
                            np.abs(v.grad[0]*v.grad[0]) +
                            v.grad[1]*v.grad[1])

def FEM(r, c, d): 

    cndty = 1*np.ones((r,c,d))
    cndty[3:25, 3:33, 10:40] = 0.001
    
    # n = x + r*(y + c*z)
    w = np.array([1 + 1*r + 25*r*c, 15 + 35*r + 25*r*c], dtype=np.longlong)
    f = np.array([0, 1], dtype=np.double)
    t1 = time.perf_counter()
    t2 = time.perf_counter()

    # try FEM on same problem
    rc = r*c
    # create tetrahedrons in lazy, inefficient way
    # First create cube mesh

    #indices to first corner of each cube
    X,Y,Z = np.meshgrid(np.arange(r-1), np.arange(c-1), np.arange(d-1), indexing='ij')
    X = X.ravel(order='F')[:,np.newaxis]
    Y = Y.ravel(order='F')[:,np.newaxis]
    Z = Z.ravel(order='F')[:,np.newaxis]
    nds = X + Y*r + Z*rc

    # defining the indices to the 8 nodes in each element
    e = np.repeat(np.array([0, 1, 1+r, r, rc, rc+1, rc+r+1, rc+r])[np.newaxis,:],
                (r-1)*(c-1)*(d-1), axis=0) + \
        np.repeat(nds, 8, axis=1)

    X,Y,Z = np.meshgrid(np.arange(r),np.arange(c), np.arange(d), indexing='ij')

    # define the skfem basis with the cubical grid to convert it into a tetrahedron basis
    m = skfem.MeshHex(np.concatenate((X.ravel(order='F')[np.newaxis,:],
                    Y.ravel(order='F')[np.newaxis,:],
                    Z.ravel(order='F')[np.newaxis,:]), axis=0), e.T )
    mtet = m.to_meshtet()
    basis = skfem.Basis(mtet, skfem.ElementTetP1())

    #Assemble the FEM matrix using custom 'form' function (you will create this)
    A = skfem.asm(form, basis, h=cndty.ravel(order='F'))
    b = np.zeros(np.size(X))

    #Apply Dirichlet conditions
    fx = np.zeros(np.size(X))
    fx[w] = f
    A,b = skfem.enforce(A, b, D=w, x=fx)

    #convert to CSC matrix and solve
    Acsc = A.tocsc()
    bc = bicg()
    x = bc.SolveA(np.size(X), np.size(X), Acsc, b, maxiter=30) # usually 1000
    phi2 = np.reshape(x, (r,c,d), order='F')

    print(f'FEM took {time.perf_counter()-t2} seconds')
    print(phi2.shape)

    return phi2

if __name__ == "__main__":
    phi2 = FEM(r,c,d)