# Import the cbcbeat module
import matplotlib.pyplot as plt
import numpy as np
from cbcbeat import *
from dolfin import *

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

def circle_heart(x,y):
    r = 0.25
    xshift = x - 0.5
    yshift = y - 0.5
    return xshift*xshift + yshift*yshift < r*r
    #return x < 5.0


#boxmesh = True
#rectanglemesh = False

boxmesh = False
rectanglemesh = True


# Define the computational domain
Nx = 40
Ny = 40
Nz = 40
time = Constant(0)


if rectanglemesh:
    mesh = RectangleMesh(Point(-0.2, -0.2), Point(1.2, 1.2), Nx, Ny)
    #mesh = RectangleMesh(Point(0, 0), Point(20, 20), Nx, Ny)
    #mesh = Mesh('pre_torso.xml')
if boxmesh:
    mesh = BoxMesh(Point(0, 0, 0), Point(20, 20, 20), Nx, Ny, Nz)


#marker = MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.domains())
# Create the submeshes
#for c in cells(mesh):
#    marker[c] = circle_heart(c.midpoint().x(), c.midpoint().y()) ## Beutel heart
#submesh = MeshView.create(marker, 2) # Heart

submesh = UnitSquareMesh(Nx, Ny)


vtkfile = File("paraview_cbcbeat_rectanglemesh/mesh.pvd")
vtkfile << submesh

# Define the conductivity (tensors)
sigma_e = 0.62                      # [Sm^-1]
sigma_i = 0.17                      # [Sm^-1]
#chi = 1400.0                       # [cm^-1]       torso mesh
#C_m = 1.0                          # [mu*F*cm−2]   torso mesh
chi = 140.0                         # [mm^-1]       fenics mesh
C_m = 0.01                          # [mu*F*mm−2]   fenics mesh
M_i = (sigma_i)/(C_m*chi)
M_e = (sigma_e)/(C_m*chi)

# Pick a cell model (see supported_cell_models for tested ones)
cell_model = FitzHughNagumoManual()


# Define some external stimulus
duration = 10                     # [ms]        torso mesh: 2-5, fenics mesh: 100
A = 50000.                       # [mu A/cm^3]
cm2mm = 10.                      # [constant]  torso mesh: 1, fenics mesh: 10
factor = 1.0/(chi*C_m)
amplitude = factor*A*(1./cm2mm)**3 # [mV/ms]

a = 11.5212; b = 13.3015; r = 1.0
S1_marker = 1
L = 0.1
S1_subdomain = CompiledSubDomain("x[0] <= L", L=L)
#S1_subdomain = CompiledSubDomain('(pow(x[0] - a, 2) + pow(x[1] - b, 2)) <= pow(r, 2)', a=a, b=b, r=r)
S1_markers = MeshFunction("size_t", submesh, submesh.topology().dim())
S1_subdomain.mark(S1_markers, S1_marker)

I_s = Expression("time >= start ? (time <= (duration + start) ? amplitude : 0.0) : 0.0",
                  time=time,
                  start=0.0,
                  duration=duration,
                  amplitude=amplitude,
                  degree=0)

stimulus = Markerwise((I_s,), (1,), S1_markers)

plot(stimulus.markers())
plt.savefig("plots/stimulus.png")
for v in stimulus.values():
    plot(v, mesh=mesh)
    plt.savefig("plots/stimulus2.png")

# Collect this information into the CardiacModel class
cardiac_model = CardiacModel(submesh, mesh, time, M_i, M_e, cell_model, stimulus, None)


# Customize and create a splitting solver
ps = SplittingSolver.default_parameters()
ps['apply_stimulus_current_to_pde'] = True
ps["theta"] = 1.0                           # Second order splitting scheme
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

N = 100
T = 750
dt = T / N

interval = (0.0, T)

if rectanglemesh:
    out_v = File("paraview_cbcbeat_rectanglemesh/bidomain_v.pvd")
    out_u = File("paraview_cbcbeat_rectanglemesh/bidomain_u.pvd")


if boxmesh:
    out_v = File("paraview_cbcbeat_boxmesh/bidomain_v.pvd")
    out_u = File("paraview_cbcbeat_boxmesh/bidomain_u.pvd")

# Solve
count = 0
v_array = np.zeros((3,N))
t = np.zeros(N)
for (timestep, fields) in solver.solve(interval, dt):
    print("(t_0, t_1) = (%g, %g)", timestep)

    """
     * "vs" (:py:class:`dolfin.Function`) representing the solution
       for the transmembrane potential and any additional statevariables
     * "vur" (:py:class:`dolfin.Function`) representing the
       transmembrane potential in combination with the extracellular
       potential and an additional Lagrange multiplier.
    """

    (vs_, vs, vur) = fields
    #t[count] = timestep[1]
    #v_array[0][count] = vs(2,10)[0]
    #v_array[1][count] = vs(10,10)[0]
    #v_array[2][count] = vs(18,10)[0]

    out_v << vur.sub(0)
    out_u << vur.sub(1)
    #vtkfile << S1_markers
    vtkfile << submesh


    count += 1


"""plt.plot(t, v_array[0], label="(2,10)")
plt.plot(t, v_array[1], label="(10,10)")
plt.plot(t, v_array[2], label="(18,10)")
plt.xlabel("t")
plt.ylabel("v")
plt.title("Transmembrane potential v at three different points")
plt.legend()
plt.savefig("plots_cbcbeat/TransmembranePlot.png")
"""

# Visualize some results
"""plt.figure()
plot(vs[0], title="Transmembrane potential (v) at end time")
c = plot(vs[0], title="Transmembrane potential (v) at end time", mode='color', vmin=-85, vmax=40)
plt.colorbar(c, orientation='vertical')
plt.savefig("TransmembranePot.png")
plt.figure()
plot(vs[-1], title="1st state variable (s_0) at end time")
plt.savefig("s_0(T).png")"""
