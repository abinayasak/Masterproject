from fenics import *
import numpy as np
import cbcbeat as c
import dolfin as df

N = 200
Nx = 50
Ny = 50
T = 500.0                       # [ms]
dt = T / N                      # [ms]
mesh = df.RectangleMesh(Point(0, 0), Point(20, 20), Nx, Ny)
#mesh = UnitSquareMesh(2, 2)
time = df.Constant(0.2)

sigma_e = 0.62                      # [Sm^-1]
sigma_i = 0.17                      # [Sm^-1]
chi = 140.                          # [mm^-1]
C_m = 0.01                          # [mu*F*mm−2]
M_i = (sigma_i)/(C_m*chi)
M_e = (sigma_e)/(C_m*chi)

I_a = 0.05


V = FunctionSpace(mesh, "P", 1)

v0 = interpolate(Expression('5*x[0]', degree=1), V)
s0 = interpolate(Expression('0', degree=0), V)

parameters = {
    "a": 0.13,
    "b": 0.013,
    "c_1": 0.26,
    "c_2": 0.1,
    "c_3": 1.0,
    "v_peak": 40.0,
    "v_rest": -85.0
}

fitzh = c.cellmodels.fitzhughnagumo_manual.FitzHughNagumoManual(params=parameters, init_conditions=None)
I = fitzh.I(-85, 0)
#I = fitzh.I(v0.vector()[:], s0.vector()[:])
#print(I)

f = fitzh.F(-85,0)
#print(f)



v_ = df.Function(V)

v_copy = v_.vector().get_local()
v_copy[v_copy < 1] = -85
v_.vector().set_local(v_copy)

print(v_)
print(v_.vector()[:])

solver = c.bidomainsolver.BasicBidomainSolver(mesh, time, M_i, M_e, I_s=None, I_a=I, v_= v_, params=None)


# Create generator
solutions = solver.solve((0.0, 1.0), 0.1)


# Iterate over generator (computes solutions as you go)
for (interval, solution_fields) in solutions:
    (t0, t1) = interval
    v_, vur = solution_fields
    # do something with the solutions
    #print(v_.vector()[:])
    print(vur.vector()[:])
