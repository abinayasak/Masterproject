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
from mshr import *
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
mesh = Mesh('pre_torso.xml')

marker = MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.domains())
heart_mesh = MeshView.create(marker, 2)
torso_mesh = MeshView.create(marker, 1)


def setup_conductivities(mesh, chi, C_m):
    # Load fibers and sheets
    Vv = VectorFunctionSpace(mesh, "DG", 0)
    fiber = Function(Vv)
    File("fiber.xml") >> fiber

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


# Define the conductivity (tensors)
"""sigma_e = 1.65                      # [Sm^-1]
sigma_i = 1.00                      # [Sm^-1]
chi = 1400                          # [cm^-1]       torso mesh
C_m = 1.0                           # [mu*F*cm−2]   torso mesh
M_i = (sigma_i)/(C_m*chi)
M_e = (sigma_e)/(C_m*chi)
M_T = 0.25*M_e
"""

chi = 400
C_m = 1.0

M_i, M_e = setup_conductivities(heart_mesh, chi, C_m)
M_T = 1.0/(C_m*chi)

# Pick a cell model (see supported_cell_models for tested ones)
#cell_model = Tentusscher_panfilov_2006_epi_cell()
cell_model = FitzHughNagumoManual()

# Define stimulus on three different areas on the torso mesh
time = Constant(0.0)
amplitude = 10

S1_subdomain_1 = CompiledSubDomain("(pow(x[0] - 11.5212, 2) + pow(x[1] - 13.3015, 2)) <= pow(0.6, 2)", degree=2)
S1_subdomain_2 = CompiledSubDomain("(pow(x[0] - 9.6885, 2) + pow(x[1] - 13.5106, 2)) <= pow(0.5, 2)", degree=2)
S1_subdomain_3 = CompiledSubDomain("(pow(x[0] - 12.5245, 2) + pow(x[1] - 15.6641, 2)) <= pow(0.6, 2)", degree=2)

S1_markers = MeshFunction("size_t", heart_mesh, heart_mesh.topology().dim())

S1_subdomain_1.mark(S1_markers, 1)
I_s1 = Expression("time >= start ? (time <= (duration + start) ? amplitude : 0.0) : 0.0",
                  time=time,
                  start=0.0,
                  duration=5.0,
                  amplitude=amplitude,
                  degree=0)

S1_subdomain_2.mark(S1_markers, 2)
I_s2 = Expression("time >= start ? (time <= (duration + start) ? amplitude : 0.0) : 0.0",
                  time=time,
                  start=0.0,
                  duration=5.0,
                  amplitude=amplitude,
                  degree=0)

S1_subdomain_3.mark(S1_markers, 3)
I_s3 = Expression("time >= start ? (time <= (duration + start) ? amplitude : 0.0) : 0.0",
                  time=time,
                  start=20.0,
                  duration=5.0,
                  amplitude=amplitude,
                  degree=0)

# Store input parameters in cardiac model
stimulus = Markerwise((I_s1,I_s2,I_s3), (1,2,3), S1_markers)

# Collect this information into the CardiacModel class
cardiac_model = CardiacModel(mesh, heart_mesh, torso_mesh, time, M_i, M_e, M_T, cell_model, stimulus)

# Customize and create a splitting solver
ps = SplittingSolver.default_parameters()
ps['apply_stimulus_current_to_pde'] = True
ps["theta"] = 0.5
ps["pde_solver"] = "bidomain"
ps["CardiacODESolver"]["scheme"] = "RL1" # 1st order Rush-Larsen for the ODEs

solver = SplittingSolver(cardiac_model, params=ps)

# Extract the solution fields and set the initial conditions
(vs_, vs, vur) = solver.solution_fields()
vs_.assign(cell_model.initial_conditions())

# Time stepping parameters
N = 500
T = 1100.
dt = T/N
print(dt)
interval = (0.0, T)


out_v = File("paraview_cbcbeat/bidomain_v.pvd")
out_u = File("paraview_cbcbeat/bidomain_u.pvd")

count = 0
u_difference = np.zeros((2,N))
t = np.zeros(N)
for (timestep, fields) in solver.solve(interval, dt):
    print("(t_0, t_1) = (%g, %g)", timestep)
    # Extract the components of the field (vs_ at previous timestep,
    # current vs, current vur)
    (vs_, vs, vur) = fields

    t[count] = timestep[1]
    u_difference[0][count] = vur.sub(1)(10,19.1) - vur.sub(1)(10,0.4)
    u_difference[1][count] = vur.sub(1)(2,10) - vur.sub(1)(25.5,10)
    print(u_difference[0][count], u_difference[1][count])
    count += 1
    out_v << vur.sub(0)
    out_u << vur.sub(1)

def plot_ECG():
    plt.figure()
    plt.plot(t, u_difference[0], "m", label="Normal: top-to-bottom")
    plt.plot(t, u_difference[1], "k", label="Normal: left-to-right")
    plt.xlabel("ms")
    plt.ylabel("mV")
    plt.title("Potential difference")
    plt.legend()
    plt.savefig("plots_cbcbeat/ecg_plot.png")

#plot_ECG()