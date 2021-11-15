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
import numpy as np

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
mesh = UnitSquareMesh(200, 200)
time = Constant(0.0)

def circle_heart(x,y):
    r = 0.25
    xshift = x - 0.5
    yshift = y - 0.5
    return xshift*xshift + yshift*yshift < r*r

marker = MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.domains())

for c in cells(mesh):
    marker[c] = circle_heart(c.midpoint().x(), c.midpoint().y()) ## Beutel heart

heart_mesh = MeshView.create(marker, 1)



mesh_cells = [mesh.num_vertices(),  mesh.num_cells()]
heart_mesh_cells = [heart_mesh.num_vertices(), heart_mesh.num_cells()]
"""
print(mesh_cells)
print(heart_mesh_cells)
print("Compute min/max cell inradius.", mesh.rmin(), mesh.rmax())
print("Compute min/max cell inradius.", heart_mesh.rmin(), heart_mesh.rmax())
"""
#[40401, 80000]
#[8049, 15718]
#Compute min/max cell inradius. 0.001464466094037834 0.0014644660940898947
#Compute min/max cell inradius. 0.0014644660940508409 0.0014644660940898622


"""
plt.figure()
plot(mesh)
plt.savefig('pictures/unitsquaremesh.png')
plt.figure()
plot(heart_mesh)
plt.savefig('pictures/circle_submesh.png')
"""


vtkfile = File("mesh/heart_mesh_unitsquaremesh.xml")
vtkfile << heart_mesh

def setup_conductivities(mesh, chi, C_m):
    # Load fibers and sheets
    Vv = VectorFunctionSpace(mesh, "DG", 0)
    fiber = Function(Vv)
    File("fibers/fiber_unitsquare.xml") >> fiber
    plt.figure()
    plot(fiber)
    plt.savefig('pictures/fiber_direction_unitsquaremesh')

    # Extract stored conductivity data.
    V = FunctionSpace(mesh, "CG", 1)

    info_blue("Using healthy conductivities")
    g_el_field = Function(V, name="g_el")
    g_et_field = Function(V, name="g_et")
    g_il_field = Function(V, name="g_il")
    g_it_field = Function(V, name="g_it")

    g_el_field.vector()[:] = 2.0/(C_m*chi)
    g_et_field.vector()[:] = 1.65/(C_m*chi)
    g_il_field.vector()[:] = 3.0/(C_m*chi)
    g_it_field.vector()[:] = 1.0/(C_m*chi)

    # Construct conductivity tensors from directions and conductivity
    # values relative to that coordinate system
    A = as_matrix([[fiber[0]], [fiber[1]]])

    from ufl import diag
    M_e_star = diag(as_vector([g_el_field, g_et_field]))
    M_i_star = diag(as_vector([g_il_field, g_it_field]))
    M_e = A*M_e_star*A.T
    M_i = A*M_i_star*A.T

    return M_i, M_e

chi = 90
C_m = 1.0
M_i, M_e = setup_conductivities(heart_mesh, chi, C_m)
M_T = 1.0/(C_m*chi)



"""
# Define the conductivity (tensors)
sigma_e = 0.62                      # [Sm^-1]
sigma_i = 0.17                      # [Sm^-1]
chi = 1400.                         # [cm^-1]
C_m = 1.0                           # [mu*F*cm^−2]
M_i = (sigma_i)/(C_m*chi)
M_e = (sigma_e)/(C_m*chi)
M_T = 1./(C_m*chi)
"""

# Pick a cell model (see supported_cell_models for tested ones)
cell_model = FitzHughNagumoManual()


S1_subdomain = CompiledSubDomain("(pow(x[0] - 0.5, 2) + pow(x[1] - 0.55, 2)) <= pow(0.15, 2)", degree=2)
S1_markers = MeshFunction("size_t", heart_mesh, heart_mesh.topology().dim())
S1_subdomain.mark(S1_markers, 1)

# Define stimulation
duration = 5.  # ms
amplitude = 10 # mV/ms
I_s = Expression("time >= start ? (time <= (duration + start) ? amplitude : 0.0) : 0.0",
                  time=time,
                  start=0.0,
                  duration=duration,
                  amplitude=amplitude,
                  degree=0)
# Store input parameters in cardiac model
stimulus = Markerwise((I_s,), (1,), S1_markers)


# Collect this information into the CardiacModel class
cardiac_model = CardiacModel(mesh, heart_mesh, time, M_i, M_e, M_T, cell_model, stimulus)


# Customize and create a splitting solver
ps = SplittingSolver.default_parameters()
ps['apply_stimulus_current_to_pde'] = True
ps["theta"] = 0.5
ps["pde_solver"] = "bidomain"
ps["CardiacODESolver"]["scheme"] = "RL1"

solver = SplittingSolver(cardiac_model, params=ps)

# Extract the solution fields and set the initial conditions
(vs_, vs, vur) = solver.solution_fields()
vs_.assign(cell_model.initial_conditions())

# Time stepping parameters
N = 3200
T = 400.
dt = T/N
interval = (0.0, T)

out_v = File("paraview_cbcbeat/bidomain_v.pvd")
out_u = File("paraview_cbcbeat/bidomain_u.pvd")

plot_figures = True
plotting_time = [25.0, 75.0, 220.0, 290.0, 360.0]
for (timestep, fields) in solver.solve(interval, dt):
    print("(t_0, t_1) = (%g, %g)", timestep)
    # Extract the components of the field (vs_ at previous timestep,
    # current vs, current vur)
    (vs_, vs, vur) = fields

    #out_v << vur.sub(0)
    #out_u << vur.sub(1)

    if plot_figures == True:
        for time in plotting_time:
            if timestep[0] == time:
                print('PLOTTING FIGURE')
                plt.figure()
                c = plot(vur.sub(0), title="v at time=%d ms" %(time), mode='color', vmin=-100, vmax=50)
                c.set_cmap("jet")
                plt.colorbar(c, orientation='vertical')
                plt.savefig("plots_cbcbeat/unitsquaremesh_v_%d.png" %(time))
                plt.figure()
                c = plot(vur.sub(1), title="u_e at time=%d ms" %(time), mode='color', vmin=-10, vmax=10)
                c.set_cmap("jet")
                plt.colorbar(c, orientation='vertical')
                plt.savefig("plots_cbcbeat/unitsquaremesh_u_e_%d.png" %(time))
