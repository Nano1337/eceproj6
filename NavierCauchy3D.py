# % Class to compute a deformation field solving the Navier-Cauchy equations for elastodynamics
# % ECE 8395: Engineering for Surgery
# % Fall 2023
# % Author: Prof. Jack Noble; jack.noble@vanderbilt.edu

from scipy import sparse
import numpy as np
from Module6_Modeling.Demo.BICG import *

def NavierCauchy(msk,w,f):
    #E = 27.04e3 # Young's modulus
    v = 0.4287 # Poisson ratio
    p1 = 1
    p2 = p1/(1-2*v)
    r,c,d = np.shape(msk)

    mskr = msk.ravel(order='F')
    rc = r * c
    N = np.array(rc*d, dtype=np.longlong) # number of nodes


    rows1 = np.repeat(np.arange(0,3*N, dtype=np.longlong)[:,np.newaxis],15,axis=1)
    ca1 = 3*np.array([0, -1, 1, -r, r, -rc, rc,
            -1-r,  -1+r,  1-r,  1+r, # xmym, xmyp, xpym, xpyp
            -1-rc, -1+rc, 1-rc, 1+rc])
    ca2 = 3*np.array([0, -r, r, -1, 1, -rc, rc,
                        -1 - r, -1 + r, 1 - r, 1 + r,  # xmym, xmyp, xpym, xpyp
                        -r - rc, -r + rc, r - rc, r + rc])
    ca3 = 3*np.array([0, -rc, rc, -r, r, -1, 1,
                        -1 - rc, -1 + rc, 1 - rc, 1 + rc,  # xmym, xmyp, xpym, xpyp
                        -r - rc, -r + rc, r - rc, r + rc])
    cols1 = rows1 + np.tile(
        np.concatenate(( (ca1 + np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]))[np.newaxis,:],#xmzm, xmzp, xpzm, xpzp
                         (ca2 + np.array([0, 0, 0, 0, 0, 0, 0,-1,-1,-1,-1, 1, 1, 1, 1]))[np.newaxis,:],
                         (ca3 + np.array([0, 0, 0, 0, 0, 0, 0,-2,-2,-2,-2,-1,-1,-1,-1]))[np.newaxis,:]), axis=0),
                (N,1))
    V1 = np.repeat(np.array([2*(3*p1+p2), -(p1+p2), -(p1+p2), -p1, -p1, -p1, -p1,
                    -p2/4, p2/4, p2/4, -p2/4, -p2/4, p2/4, p2/4, -p2/4])[np.newaxis,:], 3*N, axis=0)
    mskkp = mskr[rows1[:,0]//3] == 1
    indkp = np.where(mskkp)[0]
    cols1red = cols1[indkp,:]

    cols1redind = cols1red//3
    neumannr, neumannc = np.where(mskr[cols1redind] == 0)
    neumannnd = np.unique(cols1red[neumannr, neumannc]//3)[:,np.newaxis]

    pv = np.array([[1, 0, 0],[-1, 0, 0],[0, 1 ,0],[0, -1, 0],[ 0, 0, 1],[0, 0, -1],[
        1, 1, 0],[1 ,-1, 0],[-1, 1, 0],[-1, -1 ,0],[
        1, 0 ,1],[1, 0, -1],[-1, 0, 1],[-1, 0, -1],[
        0, 1, 1],[0, 1, -1],[ 0, -1, 1],[ 0, -1, -1]])
    pvnd = pv[:,0] + pv[:,1]*r + pv[:,2]*rc
    pvd = np.ones((18,3))
    pvd[6:,:]*=np.sqrt(2)
    rowsg = np.tile(np.concatenate((3*neumannnd, 3*neumannnd+1, 3*neumannnd+2),axis=1),(1,2))
    colsg = np.copy(rowsg)

    for i,n in enumerate(neumannnd):
        nhat = np.zeros(3)
        cnt = 0
        for j,p in enumerate(pv):
            if mskr[n + pvnd[j]]:
                nhat += p
                cnt += 1
        nhat /= cnt
        if np.sum(nhat==0)==3 or cnt==0:
            print("Invalid surface normal")
        mxj = np.argmax(np.sum(pv/pvd * np.tile(nhat[np.newaxis,:],(18,1)), axis=1))
        colsg[i,3:] = 3*(neumannnd[i] + pvnd[mxj]) + np.array([0,1,2])

    Vg = 1e6*np.ones((np.size(neumannnd),6))
    Vg[:,3:] *= -1

    mskkp2 = np.ones(np.shape(indkp), dtype=bool)
    for n in w:
        mskkp2[indkp//3 == n] = 0
    rows1u = rows1[indkp[mskkp2],:]
    cols1u = cols1red[mskkp2,:]
    V1 = V1[indkp[mskkp2], :]

    rowsps = np.concatenate(
        (rows1u[:,0], w*3, w*3+1, w*3+2,
         rowsg[:,0], rowsg[:,1], rowsg[:,2]),axis=0)
    Nn = np.size(rowsps)
    Nnd = np.shape(rows1u)[0] + 3*np.size(w)
    tbl = -np.ones(3*N + 3*np.shape(rowsps)[0], dtype=np.longlong)
    tbl[rowsps] = np.arange(0,Nn)
    rows = tbl[np.concatenate(
        (rows1u.ravel(),  w*3, w*3+1, w*3+2, rowsg.ravel())
    )]
    cols = tbl[np.concatenate(
        (cols1u.ravel(), w * 3, w * 3 + 1, w * 3 + 2, colsg.ravel())
    )]
    V = np.concatenate(
        (V1.ravel(), np.ones(np.size(w)*3), Vg.ravel())
    )
    Nn = np.size(rowsps)
    A = sparse.coo_matrix((V, (rows, cols)), shape=[Nn, Nn])
    b = np.zeros(Nn)
    b[tbl[3 * w    ]] = f[:, 0]
    b[tbl[3 * w + 1]] = f[:, 1]
    b[tbl[3 * w + 2]] = f[:, 2]

    # can also use sparse.linalg.spsolve from scipy.sparse instead of bicg
    # U = sparse.linalg.spsolve(A.tocsc(),b)
    bc = bicg()
    U = bc.SolveA(Nn, Nn, A.tocsc(), b, tol=1e-8, maxiter=10000, itol=4)
    err = np.abs(A @ U - b)

    phi = np.zeros(3*N)
    phi[rowsps] = U
    phi = np.reshape(phi,(3,r,c,d), order='F')
    return phi, err

