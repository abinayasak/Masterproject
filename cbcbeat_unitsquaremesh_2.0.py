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
mesh = UnitSquareMesh(33, 33)


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

vtkfile = File("heart_mesh_unitsquaremesh.xml")
vtkfile << heart_mesh

def setup_conductivities(mesh, chi, C_m):
    # Load fibers and sheets
    Vv = VectorFunctionSpace(mesh, "DG", 0)
    fiber = Function(Vv)
    File("fiber_unitsquaremesh.xml") >> fiber

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


"""# Define the conductivity (tensors)
sigma_e = 0.62                      # [Sm^-1]
sigma_i = 0.17                      # [Sm^-1]
chi = 2000                          # [cm^-1]       torso mesh
C_m = 1.0                           # [mu*F*cm−2]   torso mesh
M_i = (sigma_i)/(C_m*chi)
M_e = (sigma_e)/(C_m*chi)
M_T = 0.25*M_e"""

chi = 400
C_m = 1.0

M_i, M_e = setup_conductivities(heart_mesh, chi, C_m)
M_T = 1.0/(C_m*chi)


# Pick a cell model (see supported_cell_models for tested ones)
cell_model = FitzHughNagumoManual()


S1_subdomain = CompiledSubDomain("(pow(x[0] - 0.5, 2) + pow(x[1] - 0.55, 2)) <= pow(0.1, 2)", degree=2)
S1_markers = MeshFunction("size_t", heart_mesh, heart_mesh.topology().dim())
S1_subdomain.mark(S1_markers, 1)


# Define stimulation (NB: region of interest carried by the mesh
# and assumptions in cbcbeat)
duration = 5. # ms

amplitude = 10 #60 # mV/ms
I_s = Expression("time >= start ? (time <= (duration + start) ? amplitude : 0.0) : 0.0",
                  time=time,
                  start=0.0,
                  duration=duration,
                  amplitude=amplitude,
                  degree=0)
# Store input parameters in cardiac model
stimulus = Markerwise((I_s,), (1,), S1_markers)


# Collect this information into the CardiacModel class
cardiac_model = CardiacModel(mesh, heart_mesh, mesh, time, M_i, M_e, M_T, cell_model, stimulus)

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
T = 500.
dt = T/N
interval = (0.0, T)


out_v = File("paraview_cbcbeat/bidomain_v.pvd")
out_u = File("paraview_cbcbeat/bidomain_u.pvd")

time = 320
for (timestep, fields) in solver.solve(interval, dt):
    print("(t_0, t_1) = (%g, %g)", timestep)
    # Extract the components of the field (vs_ at previous timestep,
    # current vs, current vur)
    (vs_, vs, vur) = fields


    out_v << vur.sub(0)
    out_u << vur.sub(1)

    if timestep[0] == time:
        plt.figure()
        #plot(vur.sub(0), title="v at T = 600 ms")
        c = plot(vur.sub(0), title="v at time=%d ms" %(time), mode='color', vmin=-100, vmax=50)
        c.set_cmap("jet")
        plt.colorbar(c, orientation='vertical')
        plt.savefig("v_%d.png" %(time))
        plt.figure()
        #plot(vur.sub(1), title="u_e at T = 600 ms")
        c = plot(vur.sub(1), title="u_e at time=%d ms" %(time), mode='color', vmin=-100, vmax=50)
        c.set_cmap("jet")
        plt.colorbar(c, orientation='vertical')
        plt.savefig("u_e_%d.png" %(time))
