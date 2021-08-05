from dolfin import *

#### Define some functions corresponding to the steps of the Operator Splitting method ####

# Solve du/dt = L1u
def solve_L1(V, dt, theta, v_in):

    Cm = Constant(1.0) # Capacitance of the cell membrane [µF.cm-2]
    chi = Constant(2000) # area of cell membrane per unit volume [cm-1]

    A = Constant(0.04)
    v_rest = Constant(-85)# mV
    v_th = Constant(-55)# mV
    v_peak = Constant(40)# mV

    v = TrialFunction(V)
    psi = TestFunction(V)

    Ion_v = (1/chi)*v
    Ion_vin = (1/chi)*v_in

    a = v*psi*dx + dt*theta*Ion_v*psi*dx
    L = v_in*psi*dx - dt*(1-theta)*Ion_vin*psi*dx

    sol = Function(V)
    solve(a == L, sol)
    # print("[solve_L1] v min = ", sol.vector().min())
    # print("[solve_L1] v max = ", sol.vector().max())

    return sol

# Solve du/dt = L2u
def solve_L2(W, dt, theta, v_in):

    Cm = 1.0 # Capacitance of the cell membrane [µF.cm-2]
    chi =2000 # area of cell membrane per unit volume [cm-1]

    # Conductivities
    sigmai_l = 3.0 # [mS.cm-1]
    sigmai_t = 1.0 # [mS.cm-1]
    sigmai_n = 0.31525 # [mS.cm-1]
    Mi = (1/(Cm*chi))*sigmai_t

    sigmae_l = 2.0 # [mS.cm-1]
    sigmae_t = 1.65 # [mS.cm-1]
    sigmae_n = 1.3514 # [mS.cm-1]
    Me = (1/(Cm*chi))*sigmae_t

    Mt =0.25*Me

    (v,u) = TrialFunctions(W)
    (psiv,psiu) = TestFunctions(W)


    dH = Measure("dx", domain=W.sub_space(0).mesh())
    dV = Measure("dx", domain=W.sub_space(1).mesh())

    # Variational formulation
    a = v*psiv*dH + theta*dt*Mi*inner(grad(v),grad(psiv))*dH + (dt/theta)*(Mi+Me)*inner(grad(u),grad(psiu))*dV + (dt/theta)*Mt*inner(grad(u),grad(psiu))*dV + dt*Mi*inner(grad(u),grad(psiv))*dH + dt*Mi*inner(grad(v),grad(psiu))*dH
    L = v_in*psiv*dH - (1-theta)*dt*Mi*inner(grad(v_in), grad(psiv))*dH - (dt/theta)*(1-theta)*Mi*inner(grad(v_in),grad(psiu))*dH

    sol = Function(W)
    solve(a == L, sol)

    # print("[solve_L2] v min = ", sol.sub(0).vector().min())
    # print("[solve_L2] v max = ", sol.sub(0).vector().max())
    # print("[solve_L2] u min = ", sol.sub(1).vector().min())
    # print("[solve_L2] u max = ", sol.sub(1).vector().max())

    return sol

###############################################################

##### Functions for the domain / geometry #####################
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

########################### MAIN ###############################

# Global mesh = Torso + Heart
mesh = UnitSquareMesh(50, 50)
# Submesh = heart
marker = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
print("parent id = ", mesh.id())

# Create the submeshes
for c in cells(mesh):
    #marker[c] = circle_heart(c.midpoint().x(), c.midpoint().y()) ## Circle
    marker[c] = beutel_heart(c.midpoint().x(), c.midpoint().y()) ## Beutel heart

submesh_heart = MeshView.create(marker, 1) # Heart
#submesh_torso = MeshView.create(marker, 0) # Torso
print("heart id  = ", submesh_heart.id())

V = FunctionSpace(mesh, "Lagrange", 1)
H = FunctionSpace(submesh_heart, "Lagrange", 1) # Heart
#T = FunctionSpace(submesh_torso, "Lagrange", 1) # Torso

# Define the product function space
W = MixedFunctionSpace(H,V)

# Operator splitting method
# du/dt = (L1 + L2)u

# Time stepping strategy
#theta = 0 # Forward Euler
theta = 0.5 # Crank-Nicolson
#theta = 1 # Backward Euler

sol = Function(W)

# Save solution in vtk format
out_v = File("cardiac-2D-v.pvd")
out_u = File("cardiac-2D-u.pvd")

# Time-stepping
dt = 0.25
subdt = 0.25*dt
T = 10
t = 0

s_tmp = Function(W)

f = Expression("5*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.55, 2)) / 0.02)", degree=2)
s_tmp.sub(0).assign( f )

t = 0
while t <= T:
    print("t + theta*dt = ", t + theta*dt)
    subt = t
    while subt <= t + theta*dt:
        sol_s1 = solve_L1(H, subdt, theta, s_tmp.sub(0)) # Step1
        subt += subdt
        s_tmp.sub(0).assign(sol_s1)

    t = subt - subdt
    sol_s2 = solve_L2(W, dt, theta, s_tmp.sub(0)) # Step2

    while subt <= t + dt:
        sol_s3 = solve_L1(H, subdt, theta, sol_s2.sub(0)) # Step3
        subt += subdt
        sol_s2.sub(0).assign(sol_s3)

    s_tmp.sub(0).assign(sol_s3)

    sol_s2.sub(0).rename("v", "Function")
    sol_s2.sub(1).rename("u", "Function")

    out_v << sol_s2.sub(0)
    out_u << sol_s2.sub(1)

    t = subt - subdt
    print("t =", t)
