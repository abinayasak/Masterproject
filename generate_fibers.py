"""
NB: Currently only works in 2D, and lots of inefficient
hacks based on lists and loops which could be using arrays
and vectorization.
This code reads a file containing a list of
coordinate points and associated fiber vectors, and
then interpolates these to a fenics mesh according
to the algorithm from Glenn's thesis.
"""


from fenics import *
from math import sqrt,exp
import numpy as np

def read_fiber_file(filename):
    """
    Only works in 2D for now
    """
    fibers = []
    with open(filename) as infile:
        for line in infile:
            words = [float(w) for w in line.split()]
            fiber = {'pos':words[0:2],'dir':words[2:]}
            fibers.append(fiber)
    return fibers

def distance(x,y):
    #2D for now
    return sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)

def normalize(x):
    #2d and very inefficient for now
    norm = 0
    for x_ in x:
        norm += x_**2
    norm = sqrt(norm)

    for i in range(len(x)):
        x[i] = x[i]/norm

    return x

def fiber_direction(x,fiber_dict,locality=1.0):
    """
    Interpolates fiber vectors using the
    generic algorithm from Glenn's thesis.
    """

    nsd = 2
    dir = [0,0]
    for f in fiber_dict:
        dist = distance(x,f['pos'])
        w = exp(-dist*locality)
        for j in range(nsd):
            dir[j] += f['dir'][j]*w
    return normalize(dir)

fibers = read_fiber_file('fibers.txt')


if __name__ == "__main__":
    mesh = Mesh("mesh/heart_mesh.xml")
    #mesh = Mesh("mesh/heart_mesh_unitsquaremesh.xml")
    Vv = VectorFunctionSpace(mesh, "DG", 0)
    fiber = Function(Vv)

    for i, c in enumerate(Vv.tabulate_dof_coordinates()[::2]):
        #print('c=',c)
        f = fiber_direction(c,fibers,locality=1.0)
        fiber.vector()[i*2]   = f[0]
        fiber.vector()[i*2+1] = f[1]

vtkfile = File('fibers/fiber.xml')
vtkfile << fiber
