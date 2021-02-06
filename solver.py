from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
#import cbcbeat

def solver(T, N, dt, tn, Nx, Ny, degree, u0, theta):

    mesh = UnitIntervalMesh(Nx)
    V = FunctionSpace(mesh, 'P', degree)

    u = TrialFunction(V)
    v = TestFunction(V)


    #Step one
    def dvdt(v,t):
        A = 0.04       # Constant
        v_rest = 0.  # [mV]
        v_th = -65.    # [mV]
        v_peak = 1.   # [mV]
        return -A*A*(v-v_rest)*(v-v_th)*(v-v_peak)



    if tn == 0.0000:
        u0 = []
        element = V.element()
        for cell in cells(mesh):
            for i in range(element.tabulate_dof_coordinates(cell).size):
                #Discarding nodes that appears several time. Putting x coordinates into list u0
                if element.tabulate_dof_coordinates(cell)[i,0] in u0:
                    None
                else:
                    u0.append(element.tabulate_dof_coordinates(cell)[i,0])


        u0 = np.array(sorted(u0)) # Sorting the x coordinates as well as making u0 an array
        np.save("solver_array/x0", u0)
        u0[u0<0.2] = 0.           # if x < 0.2, set u0 = 0.
        u0[u0>=0.2] = -85.        # if x >= 0.2, set u0 = -85.
        np.save("solver_array/u0", u0)

        t = np.linspace(tn,tn+theta*dt,N)
        u_ode = odeint(dvdt,u0,t)
        #print("Step 1: t = [%0.4f,%0.4f]" %(tn, tn+theta*dt))
        #print(u_ode[-1,:])

    else:

        t = np.linspace(tn,tn+theta*dt,N)
        u_ode = odeint(dvdt,u0,t)
        #print("Step 1: t = [%0.4f,%0.4f]" %(tn, tn+theta*dt))
        #print(u_ode[-1,:])

    u_n = Function(V)
    u_n.vector()[:] = u_ode[-1,:]  # u from step 1, inital value for step 2



    #Step two
    M_i = 0.3
    lmda = 0.1     # M_e = lmda * M_i
    gamma = float(dt*lmda/(1 + lmda))
    if theta == 1:
        F = u*v*dx + theta*(gamma*dot(M_i*grad(u), grad(v))*dx) - u_n*v*dx
    else:
        F = u*v*dx + theta*(gamma*dot(M_i*grad(u), grad(v))*dx) - u_n*v*dx  + (1-theta)*(gamma*dot(M_i*grad(u_n), grad(v)))*dx

    a, L = lhs(F), rhs(F)

    u = Function(V) # u from step 2, inital value for step 3
    solve(a == L, u)



    #Step three
    if theta == 0.5:
        u_n = u.vector()[:]
        t = np.linspace(tn+theta*dt,tn+dt,N)
        u_ode = odeint(dvdt,u_n,t)


        u_new = Function(V)
        u_new.vector()[:] = u_ode[-1,:]

        u = u_new

    return mesh, u


def run_solver():

    theta = 1 # =0.5 Strang/CN , =1 Godunov/BE
    N = 100
    Nx = 400; Ny = 8
    T = 1.    #[s]
    dt = T/N  #[s]
    degree = 2
    u0 = None


    tn = 0
    for i in range(N+1):
        print("tn: %0.4f / %0.4f" %(tn, T))
        mesh, u = solver(T, N, dt, tn, Nx, Ny, degree, u0, theta)
        tn += dt

        plot(u)
        plot(mesh)

        #vtkfile = File('solver/solution%d.pvd' %i)
        #vtkfile << u

        plt.savefig ("solver_plot/test%d.png" %i)

        np.save("solver_array/u%d" % (i+1), u.vector()[:])

        u0 = u.vector()[:]






if __name__ == '__main__':
    run_solver()
