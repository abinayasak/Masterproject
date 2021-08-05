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
mesh = UnitSquareMesh(50, 50)
mesh = UnitSquareMesh(50, 50)
time = Constant(0.0)

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
stimulus = Expression('x[0] <= 0.2 ? 0 : -85', degree=0)

# Collect this information into the CardiacModel class
cardiac_model = CardiacModel(mesh, mesh, time, M_i, M_e, cell_model, stimulus)

# Customize and create a splitting solver
ps = SplittingSolver.default_parameters()
ps["theta"] = 1.0                        # Second order splitting scheme
ps["pde_solver"] = "bidomain"          # Use Monodomain model for the PDEs
ps["CardiacODESolver"]["scheme"] = "RL1" # 1st order Rush-Larsen for the ODEs
ps["BidomainSolver"]["linear_solver_type"] = "iterative"
ps["BidomainSolver"]["algorithm"] = "cg"
ps["BidomainSolver"]["preconditioner"] = "petsc_amg"
ps['apply_stimulus_current_to_pde'] = True

solver = SplittingSolver(cardiac_model, params=ps)

# Extract the solution fields and set the initial conditions
(vs_, vs, vur) = solver.solution_fields()
vs_.assign(cell_model.initial_conditions())

# Time stepping parameters
N = 100 #1000 #2000
T = 10 #300 #500
dt = T / N
interval = (0.0, T)


# Solve!
count = 0
v_array = np.zeros((3,N))
for (timestep, fields) in solver.solve(interval, dt):
    #print("(t_0, t_1) = (%g, %g)", timestep)

    # Extract the components of the field (vs_ at previous timestep,
    # current vs, current vur)
    (vs_, vs, vur) = fields
    v_array[0][count] = vs(0.2,0.5)[0]
    v_array[1][count] = vs(0.5,0.5)[0]
    v_array[2][count] = vs(0.8,0.5)[0]
    count += 1

t = np.linspace(0, T, N)
plt.plot(t, v_array[0], label="(0.2,0.5)")
plt.plot(t, v_array[1], label="(0.5,0.5)")
plt.plot(t, v_array[2], label="(0.8,0.5)")
plt.xlabel("t")
plt.ylabel("v")
plt.title("Transmembrane potential v at three different points")
plt.legend()
plt.savefig("plots_cbcbeat/TransmembranePlot_demo.png")


# Visualize some results
plt.figure()
plot(vs[0], title="Transmembrane potential (v) at end time")
c = plot(vs[0], title="Transmembrane potential (v) at end time", mode='color', vmin=-85, vmax=150)
plt.colorbar(c, orientation='vertical')
plt.savefig("TransmembranePot.png")
plt.figure()
plot(vs[-1], title="1st state variable (s_0) at end time")
plt.savefig("s_0(T).png")
