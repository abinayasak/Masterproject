#!/usr/bin/env python
#  -*- coding: utf-8 -*-

# .. _first_example
#
# A basic practical example of how to use the cbcbeat module, in
# particular how to solve the monodomain equations coupled to a
# moderately complex cell model using the splitting solver provided by
# cbcbeat.
#
# How to use the cbcbeat module to solve a cardiac EP problem
# ===========================================================
#
# This demo shows how to
# * Use a cardiac cell model from supported cell models
# * Define a cardiac model based on a mesh and other input
# * Use and customize the main solver (SplittingSolver)

# Import the cbcbeat module
import matplotlib.pyplot as plt
import numpy as np
from cbcbeat import *
from dolfin import *
import dolfin as df
#print (df.__version__)

# Turn on FFC/FEniCS optimizations
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
parameters["form_compiler"]["quadrature_degree"] = 3

# Turn off adjoint functionality
import cbcbeat
if cbcbeat.dolfin_adjoint:
    parameters["adjoint"]["stop_annotating"] = True

# Define the computational domain
Nx = 50
Ny = 50
#mesh = RectangleMesh(Point(0, 0), Point(20, 20), Nx, Ny)
time = Constant(0.0)

def circle_heart(x,y):
    r = 0.25
    xshift = x - 0.5
    yshift = y - 0.5
    return xshift*xshift + yshift*yshift < r*r

def beutel_heart(x,y):
    a = 0.05
    xshift = x - 0.5
    yshift = y - 0.5
    return (xshift*xshift + yshift*yshift - a)*(xshift*xshift + yshift*yshift - a)*(xshift*xshift + yshift*yshift - a) < xshift*xshift*yshift*yshift*yshift


mesh = UnitSquareMesh(Nx, Ny)
marker = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)

# Create the submeshes
for c in cells(mesh):
    marker[c] = circle_heart(c.midpoint().x(), c.midpoint().y()) ## Beutel heart

submesh_heart = MeshView.create(marker, 1) # Heart


# Define the conductivity (tensors)
sigma_e = 0.62                      # [Sm^-1]
sigma_i = 0.17                      # [Sm^-1]
chi = 140.                          # [mm^-1]
C_m = 0.01                          # [mu*F*mm−2]
M_i = (sigma_i)/(C_m*chi)
M_e = (sigma_e)/(C_m*chi)

# Pick a cell model (see supported_cell_models for tested ones)
cell_model = FitzHughNagumoManual()


# Define some external stimulus
#stimulus = Expression('x[0] <= 2.0 ? 0 : -85', t=time, degree=1)
stimulus = Expression("10*t*x[0]", t=time, degree=1)

# Collect this information into the CardiacModel class
cardiac_model = CardiacModel(submesh_heart, time, M_i, M_e, cell_model, stimulus)


# Customize and create a splitting solver
ps = SplittingSolver.default_parameters()
ps["theta"] = 0.5                            # Second order splitting scheme
ps["pde_solver"] = "bidomain"                # Use Bidomain model for the PDEs
ps["CardiacODESolver"]["scheme"] = "RL1"     # 1st order Rush-Larsen for the ODEs
ps["BidomainSolver"]["linear_solver_type"] = "iterative"
ps["BidomainSolver"]["algorithm"] = "cg"
ps["BidomainSolver"]["preconditioner"] = "petsc_amg"

#info(ps, True)
solver = SplittingSolver(cardiac_model, params=ps)

# Extract the solution fields and set the initial conditions
(vs_, vs, vur) = solver.solution_fields()
vs_.assign(cell_model.initial_conditions())

# Time stepping parameters

dt = 0.1
T = 1.0
#N = 1000
#T = 100.0
#dt = T / N
interval = (0.0, T)

timer = Timer("XXX Forward solve") # Time the total solve

# Solve!
for (timestep, fields) in solver.solve(interval, dt):
    print("(t_0, t_1) = (%g, %g)", timestep)

    # Extract the components of the field (vs_ at previous timestep,
    # current vs, current vur)
    (vs_, vs, vur) = fields

timer.stop()

# Visualize some results
plt.figure()
plot(vs[0], title="Transmembrane potential (v) at end time")
plt.savefig("TransmembranePot.png")
plt.figure()
plot(vs[-1], title="1st state variable (s_0) at end time")
plt.savefig("s_0(T).png")
# List times spent
#list_timings(TimingClear.keep, [TimingType.user])

#print("Success!")
