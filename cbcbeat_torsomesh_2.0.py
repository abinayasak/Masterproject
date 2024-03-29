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
mesh = Mesh('mesh/pre_torso.xml')
marker = MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.domains())

refined_mesh = mesh
refined_marker = marker

#refined_mesh = adapt(mesh)
#refined_marker = adapt(marker, refined_mesh)

heart_mesh = MeshView.create(refined_marker, 2)
torso_mesh = MeshView.create(refined_marker, 1)

#refined_mesh = adapt(refined_mesh)
#refined_marker = adapt(refined_marker, refined_mesh)
#heart_mesh = MeshView.create(refined_marker, 2)


"""
plt.figure()
plot(refined_mesh)
plt.savefig('pictures/refined_mesh.png')
plt.figure()
plot(heart_mesh)
plt.savefig('pictures/heart_mesh.png')


plot(refined_mesh)
plt.plot(12,18.3,'ro')
plt.plot(12,0.3, 'bo')
plt.plot(1.6,13.5, 'ro')
plt.plot(25.8,13.5, 'bo')
plt.savefig('pictures/surface_potential_points.png')
"""

"""
mesh_cells = [refined_mesh.num_vertices(),  refined_mesh.num_cells()]
heart_mesh_cells = [heart_mesh.num_vertices(), heart_mesh.num_cells()]
print(mesh_cells)
print(heart_mesh_cells)
print("Compute min/max cell inradius.", refined_mesh.rmin(), refined_mesh.rmax())
print("Compute min/max cell inradius.", heart_mesh.rmin(), heart_mesh.rmax())
"""
vtkfile = File('mesh/heart_mesh_torsomesh.xml')
vtkfile << heart_mesh



def setup_conductivities(mesh, chi, C_m):
    # Load fibers and sheets
    Vv = VectorFunctionSpace(mesh, "DG", 0)
    fiber = Function(Vv)
    File("fibers/fiber_torso.xml") >> fiber
    plt.figure()
    plot(fiber)
    plt.savefig('pictures/fiber_direction')

    # Extract stored conductivity data.
    V = FunctionSpace(mesh, "CG", 1)

    info_blue("Using healthy conductivities")
    g_el_field = Function(V, name="g_el")
    g_et_field = Function(V, name="g_et")
    g_il_field = Function(V, name="g_il")
    g_it_field = Function(V, name="g_it")

    g_el_field.vector()[:] = 2.0/(C_m*chi) #2.0/(C_m*chi)
    g_et_field.vector()[:] = 1.65/(C_m*chi) #1.65/(C_m*chi)
    g_il_field.vector()[:] = 3.0/(C_m*chi) #3.0/(C_m*chi)
    g_it_field.vector()[:] = 1.0/(C_m*chi) #1.0/(C_m*chi)

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


cell_model = Tentusscher_panfilov_2006_epi_cell()
#cell_model = FitzHughNagumoManual()

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
cardiac_model = CardiacModel(refined_mesh, heart_mesh, time, M_i, M_e, M_T, cell_model, stimulus)

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
N = 200
T = 400
dt = T/N
interval = (0.0, T)

out_v = File("paraview_cbcbeat/bidomain_v.pvd")
out_u = File("paraview_cbcbeat/bidomain_u.pvd")


count = 0
u_difference = np.zeros((2,N))
t = np.zeros(N)
action_potential = np.zeros(N)
plot_figures = True
plotting_time = [25.0, 75.0, 220.0, 290.0, 360.0]
for (timestep, fields) in solver.solve(interval, dt):
    print("(t_0, t_1) = (%g, %g)", timestep)
    # Extract the components of the field (vs_ at previous timestep,
    # current vs, current vur)
    (vs_, vs, vur) = fields


    action_potential[count] = vur.sub(0)(12,14)
    t[count] = timestep[1]
    u_difference[0][count] = vur.sub(1)(10,19) - vur.sub(1)(10,0.3)
    u_difference[1][count] = vur.sub(1)(1.6,10) - vur.sub(1)(25.8,10)

    count += 1
    out_v << vur.sub(0)
    out_u << vur.sub(1)

    if plot_figures == True:
        for time in plotting_time:
            if timestep[0] == time:
                print('PLOTTING FIGURE')
                plt.figure()
                c = plot(vur.sub(0), title="v at time=%d ms" %(time), mode='color', vmin=-100, vmax=50)
                c.set_cmap("jet")
                plt.colorbar(c, fraction=0.043, pad=0.009)
                plt.savefig("plots_cbcbeat/torsomesh_v_%d.png" %(time))
                plt.figure()
                c = plot(vur.sub(1), title="u_e at time=%d ms" %(time), mode='color', vmin=-10, vmax=10)
                c.set_cmap("jet")
                plt.colorbar(c, fraction=0.034, pad=0.009)
                plt.savefig("plots_cbcbeat/torsomesh_u_e_%d.png" %(time))


np.save("u_difference.npy", u_difference)
np.save("action_potential", action_potential)
np.save("t", t)

def plot_ECG():
    plt.figure()
    plt.plot(t, u_difference[0], "m", label="top-to-bottom")
    plt.plot(t, u_difference[1], "k", label="left-to-right")
    plt.xlabel("ms")
    plt.ylabel("mV")
    plt.title("Surface potential difference")
    plt.legend()
    plt.savefig("plots_cbcbeat/surface_potential.png")



plot_ECG()
