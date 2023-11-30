# % Project 6 driver
# % ECE 8395: Engineering for Surgery
# % Fall 2023
# % Author: Prof. Jack Noble; jack.noble@vanderbilt.edu

import json
import scipy
import numpy as np
from NavierCauchy3D import *
from Module2_ImageProc.Demo.DisplayVolume import *
from skimage import measure
from scipy.interpolate import interpn
import vtk
# You need to create this:
from TrackerPointDeform import *

def ActorDecorator(func):
    def inner(verts, faces = None, color=[1,0,0], opacity=1.0):
        pnts = vtk.vtkPoints()
        for j,p in enumerate(verts):
            pnts.InsertPoint(j,p)

        poly = func(pnts, faces)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color[0], color[1], color[2])
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetPointSize(6)
        return actor
    return inner

@ActorDecorator
def surfaceActor(pnts, faces):
    cells = vtk.vtkCellArray()
    for j in range(len(faces)):
        vil = vtk.vtkIdList()
        for k in range(3):
            vil.InsertNextId(faces[j,k])
        cells.InsertNextCell(vil)

    poly = vtk.vtkPolyData()
    poly.SetPoints(pnts)
    poly.SetPolys(cells)

    poly.BuildCells()
    poly.BuildLinks()
    return poly

@ActorDecorator
def pointActor(pnts, faces = None):
    cells = vtk.vtkCellArray()
    for j in range(pnts.GetNumberOfPoints()):
        vil = vtk.vtkIdList()
        vil.InsertNextId(j)
        cells.InsertNextCell(vil)

    poly = vtk.vtkPolyData()
    poly.SetPoints(pnts)
    poly.SetVerts(cells)

    return poly

# if additional Dirichlet nodes are desired they should be input in src2 as
# an [N,1] vector of node indices to these Dirichlet nodes. src2vect should
# be an [N,3] array containing the displacement vector for each
# cooresponding Dirichlet node
def project6_Driver(src2 = None, src2vect = None):
    f = open('Project6_driver.json','rt')
    dt = json.load(f)
    voxsz = np.array(dt['lvr']['voxsz'])
    lvr = np.array(dt['lvr']['data'])
    liver_landmarks = np.array(dt['liver_landmarks'])
    liver_clamp_landmarks = np.array(dt['liver_clamp_landmarks'])
    src1 = np.array(dt['src1'])
    vect1 = np.array(dt['vect1'])
    Tt1_f2 = np.array(dt['Tt1_f2'])
    r,c,d = np.shape(lvr)

    # initial run of NavierCauchy to flatten the liver as it rests on the table
    phi, err = NavierCauchy(lvr, src1, vect1)
    # Second run is deformation that accounts for any other input Dirichlet conditions
    if src2 is not None:
        phi2, err2 = NavierCauchy(lvr, np.concatenate((src1, src2)),
                                  np.concatenate((vect1, src2vect), axis=0))
    else:
        phi2 = np.zeros(np.size(phi))

    # deform a liver surface using the flattening transformation
    vertices, triangles, dc1, dc2 = measure.marching_cubes(lvr, 0.5)
    xs, ys, zs = (np.arange(0, r), np.arange(0, c), np.arange(0, d))
    dx = interpn((xs, ys, zs), np.squeeze(phi[0, :, :, :]), vertices, 'linear', False, 0)
    dy = interpn((xs, ys, zs), np.squeeze(phi[1, :, :, :]), vertices, 'linear', False, 0)
    dz = interpn((xs, ys, zs), np.squeeze(phi[2, :, :, :]), vertices, 'linear', False, 0)
    liverflat = vertices + np.concatenate((dx[:, np.newaxis], dy[:, np.newaxis], dz[:, np.newaxis]), axis=1)

    # map optical tracker coordinates to flattened liver image coordinates using student function
    p = trackerPointDeform(liver_landmarks.T, Tt1_f2, np.zeros(np.shape(phi2)), voxsz)
    p2 = trackerPointDeform(liver_clamp_landmarks.T, Tt1_f2, np.zeros(np.shape(phi2)), voxsz)
    # Display the x component of the deformation field,
    dis = DisplayVolume()
    dis.SetImage(np.squeeze(phi[0, :, :, :]), voxsz, level=-0, contrast=3, interpolation='nearest')


    # pre-clamp optical tracker probe points in blue, post clamp probe points in red, the flattened liver surface in green
    renwin = vtk.vtkRenderWindow()
    ren = vtk.vtkRenderer()
    renwin.AddRenderer(ren)
    inter = vtk.vtkRenderWindowInteractor()
    inter.SetRenderWindow(renwin)
    inter.Initialize()
    inter.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    ren.AddActor(surfaceActor(liverflat*voxsz[np.newaxis,:], triangles, [0,1,0], 0.25))
    ren.AddActor(pointActor(p.T,color=[0,0,1]))
    ren.AddActor(pointActor(p2.T,color=[1,0,0]))


    # Display the navier-cauchy estimated liver surface after clamp deformation in red, and predicted landmarks in green
    dx2 = interpn((xs, ys, zs), np.squeeze(phi2[0, :, :, :]), vertices, 'linear', False, 0)
    dy2 = interpn((xs, ys, zs), np.squeeze(phi2[1, :, :, :]), vertices, 'linear', False, 0)
    dz2 = interpn((xs, ys, zs), np.squeeze(phi2[2, :, :, :]), vertices, 'linear', False, 0)
    liverclamp = vertices + np.concatenate((dx2[:, np.newaxis], dy2[:, np.newaxis], dz2[:, np.newaxis]), axis=1)

    ren.AddActor(surfaceActor(liverclamp * voxsz[np.newaxis,:],triangles,[1,0,0],0.5))

    pdef = trackerPointDeform(liver_landmarks.T, Tt1_f2, phi2 - phi, voxsz)
    ren.AddActor(pointActor(pdef.T,color=[0,1,0]))
    # Here, compute RMS error between post clamp probe points and pre clamp probe points (which does not account for deformation due to clamp)

    # Here, compute RMS error between post clamp probe points and predicted landmark positions created using Navier-Cauchy
    renwin.Render()
    return p, p2, pdef, renwin
