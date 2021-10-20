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
from cbcbeat import *

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
mesh = UnitSquareMesh(50, 50)
time = Constant(0.0)

def circle_heart(x,y):
    r = 0.25
    xshift = x - 0.5
    yshift = y - 0.5
    return xshift*xshift + yshift*yshift < r*r

marker = MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.domains())

for c in cells(mesh):
    marker[c] = circle_heart(c.midpoint().x(), c.midpoint().y()) ## Beutel heart

submesh = MeshView.create(marker, 1)


# Define the conductivity (tensors)
sigma_e = 0.62                      # [Sm^-1]
sigma_i = 0.17                      # [Sm^-1]
chi = 1400                           # [cm^-1]       torso mesh
C_m = 1.0                          # [mu*F*cm−2]   torso mesh
M_i = (sigma_i)/(C_m*chi)
M_e = (sigma_e)/(C_m*chi)

# Pick a cell model (see supported_cell_models for tested ones)
cell_model = FitzHughNagumoManual()


S1_subdomain = CompiledSubDomain("(pow(x[0] - 0.5, 2) + pow(x[1] - 0.55, 2)) <= pow(0.1, 2)", degree=2)
S1_markers = MeshFunction("size_t", submesh, submesh.topology().dim())
S1_subdomain.mark(S1_markers, 1)


# Define stimulation (NB: region of interest carried by the mesh
# and assumptions in cbcbeat)
duration = 2. # ms

amplitude = 37 # mV/ms
I_s = Expression("time >= start ? (time <= (duration + start) ? amplitude : 0.0) : 0.0",
                  time=time,
                  start=0.0,
                  duration=duration,
                  amplitude=amplitude,
                  degree=0)
# Store input parameters in cardiac model
stimulus = Markerwise((I_s,), (1,), S1_markers)


# Collect this information into the CardiacModel class
cardiac_model = CardiacModel(mesh, submesh, time, M_i, M_e, cell_model, stimulus)

# Customize and create a splitting solver
ps = SplittingSolver.default_parameters()
ps['apply_stimulus_current_to_pde'] = True
ps["theta"] = 1.0                        # Second order splitting scheme
ps["pde_solver"] = "bidomain"          # Use Monodomain model for the PDEs
ps["CardiacODESolver"]["scheme"] = "RL1" # 1st order Rush-Larsen for the ODEs
ps["BidomainSolver"]["linear_solver_type"] = "iterative"
ps["BidomainSolver"]["algorithm"] = "cg"
ps["BidomainSolver"]["preconditioner"] = "petsc_amg"

solver = SplittingSolver(cardiac_model, params=ps)

# Extract the solution fields and set the initial conditions
(vs_, vs, vur) = solver.solution_fields()
vs_.assign(cell_model.initial_conditions())

# Time stepping parameters
N = 200
T = 600.
dt = T/N
interval = (0.0, T)


out_v = File("paraview_cbcbeat/bidomain_v.pvd")
out_u = File("paraview_cbcbeat/bidomain_u.pvd")


for (timestep, fields) in solver.solve(interval, dt):
    print("(t_0, t_1) = (%g, %g)", timestep)

    # Extract the components of the field (vs_ at previous timestep,
    # current vs, current vur)
    (vs_, vs, vur) = fields


    out_v << vur.sub(0)
    out_u << vur.sub(1)
